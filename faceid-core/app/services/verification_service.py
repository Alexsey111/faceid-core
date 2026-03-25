# app/services/verification_service.py - Сервис верификации

import logging
from typing import Dict, Any, Optional
import numpy as np
import time

from app.ml.pipeline import FacePipeline
from app.ml.pipeline_v2 import FacePipelineV2
from app.core.config import settings
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.verification_repo import VerificationRepository
from app.services.liveness_service import LivenessService
from app.services.anti_replay_service import AntiReplayService
from app.services.search_service import SearchService
try:
    from app.monitoring.metrics import (
        VERIFY_LATENCY,
        VERIFY_RESULT_COUNTER,
        LIVENESS_RESULT_COUNTER,
        LIVENESS_FAIL_COUNT,
        LIVENESS_MS,
        PIPELINE_STAGE_DURATION,
        PIPELINE_MS,
        DETECT_MS,
        ENCODE_MS,
    )
    METRICS_ENABLED = True
except Exception:
    METRICS_ENABLED = False

logger = logging.getLogger("verification")


def _metric_bool(value: Optional[bool]) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "none"


def _log_extra(job_id: Optional[str], **fields: Any) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if job_id is not None:
        extra["job_id"] = job_id
    extra.update(fields)
    return extra


def _observe_stage(stage: str, duration_ms: float) -> None:
    if METRICS_ENABLED:
        try:
            PIPELINE_STAGE_DURATION.labels(stage=stage).observe(duration_ms)
        except Exception:
            pass


def _observe_pipeline_metrics(pipeline_time: dict[str, Any]) -> None:
    if not METRICS_ENABLED:
        return

    try:
        if "total_pipeline_ms" in pipeline_time:
            PIPELINE_MS.observe(float(pipeline_time["total_pipeline_ms"]))
        detect_ms = 0.0
        if "detect_ms" in pipeline_time:
            detect_ms = float(pipeline_time["detect_ms"])
        else:
            detect_ms += float(pipeline_time.get("fast_detect_ms", 0.0))
            detect_ms += float(pipeline_time.get("fallback_detect_ms", 0.0))
        if detect_ms > 0.0:
            DETECT_MS.observe(detect_ms)
        if "encode_ms" in pipeline_time:
            ENCODE_MS.observe(float(pipeline_time["encode_ms"]))
        if "liveness_ms" in pipeline_time:
            LIVENESS_MS.observe(float(pipeline_time["liveness_ms"]))
    except Exception:
        pass


def _observe_verify_latency(start_time: float) -> None:
    if METRICS_ENABLED:
        try:
            latency = (time.time() - start_time) * 1000
            VERIFY_LATENCY.observe(latency)
        except Exception:
            pass


def _record_verify_result(result: str) -> None:
    if METRICS_ENABLED:
        try:
            VERIFY_RESULT_COUNTER.labels(result=result).inc()
        except Exception:
            pass


def _record_liveness_result(passed: Optional[bool]) -> None:
    if METRICS_ENABLED:
        try:
            value = _metric_bool(passed)
            LIVENESS_RESULT_COUNTER.labels(result=value).inc()
        except Exception:
            pass


class VerificationService:

    def __init__(
        self,
        embedding_repo: EmbeddingRepository | None,
        verification_repo: VerificationRepository | None,
        search_service: SearchService | None = None,
        pipeline: Any | None = None,
    ):
        self.embedding_repo = embedding_repo
        self.verification_repo = verification_repo
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            self.pipeline = FacePipelineV2() if settings.USE_PIPELINE_V2 else FacePipeline()
        self.search_service = search_service
        if self.search_service is None and embedding_repo is not None:
            self.search_service = SearchService(embedding_repo)

    def extract_features(self, image_bytes: bytes) -> dict:
        result = self.pipeline.process(image_bytes)

        if result.get("status") == "spoof":
            liveness_score = result.get("liveness_score", result.get("confidence"))
            LIVENESS_FAIL_COUNT.inc()
            return {
                "status": "spoof",
                "liveness_passed": False,
                "liveness": {
                    "score": liveness_score,
                    "risk": "spoof",
                },
                "timings": result.get("timings", {}),
            }

        embedding = np.asarray(result["embedding"], dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0.0:
            raise ValueError("Invalid embedding vector")

        embedding = embedding / norm

        liveness = result.get("liveness", {})

        return {
            "embedding": embedding,
            "liveness": liveness,
            "timings": result.get("timings", {}),
        }

    def make_decision(
        self,
        embedding,
        top_k,
        liveness,
        user_id=None,
    ) -> dict:
        _ = embedding
        _ = liveness
        _ = user_id

        if not top_k:
            return {"status": "no_match"}

        similarity = float(top_k[0]["similarity"])

        if similarity >= settings.HIGH_THRESHOLD:
            return {
                "status": "match",
                "user_id": top_k[0]["user_id"],
                "similarity": similarity,
            }

        if similarity <= settings.LOW_THRESHOLD:
            return {
                "status": "no_match",
                "similarity": similarity,
            }

        return {
            "status": "low_confidence",
            "similarity": similarity,
        }

    async def verify_face(
        self,
        image_bytes: bytes,
        user_id: Optional[str] = None,
        require_liveness: bool = False,
        check_replay: bool = True,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._verify_face_impl(
            image_bytes=image_bytes,
            user_id=user_id,
            require_liveness=require_liveness,
            check_replay=check_replay,
            job_id=job_id,
        )

    async def _verify_face_impl(
        self,
        image_bytes: bytes,
        user_id: Optional[str] = None,
        require_liveness: bool = False,
        check_replay: bool = True,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        t_start = time.time()

        verification_repo = self.verification_repo
        if verification_repo is None:
            raise RuntimeError("VerificationRepository is required for verify_face")

        search_service = self.search_service
        if search_service is None:
            raise RuntimeError("SearchService is required for verify_face")

        embedding_repo = self.embedding_repo
        if embedding_repo is None:
            raise RuntimeError("EmbeddingRepository is required for verify_face")

        # Convert user_id to int or None
        try:
            user_id_int: Optional[int] = int(user_id) if user_id else None
        except (ValueError, TypeError):
            user_id_int = None

        # ANTI-REPLAY CHECK (just a signal, don't block)
        replay_detected = False
        if check_replay:
            try:
                replay_detected = not AntiReplayService.check(image_bytes)
            except Exception as exc:
                logger.warning(
                    "anti_replay_unavailable job_id=%s error=%s",
                    job_id,
                    exc,
                )
                replay_detected = False

        logger.info(
            "verify_started job_id=%s",
            job_id,
            extra=_log_extra(
                job_id,
                user_id=user_id,
                require_liveness=require_liveness,
            ),
        )

        try:
            features = self.extract_features(image_bytes)
        except ValueError as e:
            if str(e) == "Invalid embedding vector":
                _record_verify_result("no_match")
                _observe_verify_latency(t_start)
                return {
                    "status": "no_match",
                    "similarity": 0.0,
                    "liveness_passed": None,
                    "replay_detected": replay_detected,
                }

            logger.exception(
                "pipeline_failed job_id=%s",
                job_id,
                extra=_log_extra(job_id, error=str(e)),
            )

            await verification_repo.create_log(
                user_id=user_id_int,
                similarity=0.0,
                success=False,
                margin=None,
                liveness_score=None,
                is_genuine=None
            )
            _record_verify_result("processing_failed")
            _observe_verify_latency(t_start)

            return {
                "status": "processing_failed",
                "liveness_passed": False,
                "replay_detected": replay_detected
            }
        except Exception as e:
            logger.exception(
                "pipeline_failed job_id=%s",
                job_id,
                extra=_log_extra(job_id, error=str(e)),
            )

            await verification_repo.create_log(
                user_id=user_id_int,
                similarity=0.0,
                success=False,
                margin=None,
                liveness_score=None,
                is_genuine=None
            )
            _record_verify_result("processing_failed")
            _observe_verify_latency(t_start)

            return {
                "status": "processing_failed",
                "liveness_passed": False,
                "replay_detected": replay_detected
            }

        if features.get("status") == "spoof":
            liveness = features["liveness"]
            liveness_score = float(liveness.get("score", 0.0))
            liveness_risk = liveness.get("risk", "spoof")

            await verification_repo.create_log(
                user_id=user_id_int,
                similarity=0.0,
                success=False,
                margin=None,
                liveness_score=liveness_score,
                is_genuine=None
            )
            _record_verify_result("spoof_detected")
            _record_liveness_result(False)
            LIVENESS_FAIL_COUNT.inc()
            _observe_verify_latency(t_start)

            return {
                "status": "spoof",
                "liveness_passed": False,
                "liveness": {
                    "score": liveness_score,
                    "risk": liveness_risk,
                },
                "replay_detected": replay_detected,
            }

        embedding = features["embedding"]
        liveness_signals = features["liveness"]
        pipeline_time = features["timings"]

        _observe_pipeline_metrics(pipeline_time)

        if "total_pipeline_ms" in pipeline_time:
            _observe_stage("pipeline", float(pipeline_time["total_pipeline_ms"]))
        if "detect_ms" in pipeline_time:
            _observe_stage("detect", float(pipeline_time["detect_ms"]))
        if "encode_ms" in pipeline_time:
            _observe_stage("embed", float(pipeline_time["encode_ms"]))
        if "liveness_ms" in pipeline_time:
            _observe_stage("liveness", float(pipeline_time["liveness_ms"]))

        logger.info(
            "pipeline_completed job_id=%s",
            job_id,
            extra=_log_extra(job_id, timings=pipeline_time),
        )

        # Compute composite liveness score
        liveness_result = LivenessService.fuse(liveness_signals)
        liveness_score = liveness_result["score"]
        liveness_risk = liveness_result["risk"]

        # LIVENESS CHECK
        liveness_passed = LivenessService.is_passed(liveness_signals)
        if require_liveness and not liveness_passed:

            await verification_repo.create_log(
                user_id=user_id_int,
                similarity=0.0,
                success=False,
                margin=None,
                liveness_score=liveness_score,
                is_genuine=None
            )
            _record_verify_result("spoof_detected")
            _record_liveness_result(False)
            LIVENESS_FAIL_COUNT.inc()
            _observe_verify_latency(t_start)

            return {
                "status": "spoof",
                "liveness_passed": False,
                "liveness": {
                    "score": liveness_score,
                    "risk": liveness_risk
                },
                "replay_detected": replay_detected
            }

        # VECTOR SEARCH (top-2 for margin calculation)
        t0 = time.time()
        top_k = await search_service.search_top_k(embedding, k=2)
        search_time = (time.time() - t0) * 1000
        _observe_stage("search", search_time)

        top1_similarity = top_k[0]["similarity"] if top_k else 0.0
        top2_similarity = top_k[1]["similarity"] if len(top_k) > 1 else 0.0
        margin = top1_similarity - top2_similarity
        logger.info(
            "search_completed job_id=%s",
            job_id,
            extra=_log_extra(
                job_id,
                top1_similarity=top1_similarity,
                top2_similarity=top2_similarity,
                margin=margin,
                search_ms=search_time,
            ),
        )

        similarity = top1_similarity
        matched_user_id = top_k[0]["user_id"] if top_k else None

        # Check if matches requested user_id (by external_id)
        is_genuine: bool | None = None
        if user_id:
            from app.db.repositories.user_repo import UserRepository
            user_repo = UserRepository(embedding_repo.db)
            expected_user = await user_repo.get_by_external_id(user_id)

            if expected_user:
                expected_user_id = getattr(expected_user, "id")

                # Get all embeddings for this user and find best score
                embeddings = await search_service.search_user_embeddings(expected_user_id)
                if embeddings:
                    embedding_vectors = np.stack(embeddings)
                    centroid = np.mean(embedding_vectors, axis=0)
                    centroid_norm = np.linalg.norm(centroid)
                    if centroid_norm != 0:
                        centroid = centroid / centroid_norm
                        similarity = float(np.dot(embedding, centroid))
                    else:
                        similarity = 0.0
                    matched_user_id = expected_user_id
                    is_genuine = True
                else:
                    matched_user_id = expected_user_id
                    is_genuine = False
                    similarity = 0.0

        decision = self.make_decision(embedding, top_k, liveness_signals, user_id=user_id)
        decision_status = decision["status"]
        decision_similarity = float(decision.get("similarity", similarity))
        decision_user_id = decision.get("user_id", matched_user_id)

        await verification_repo.create_log(
            user_id=decision_user_id,
            similarity=decision_similarity,
            margin=margin,
            liveness_score=liveness_score,
            success=(decision_status == "match"),
            is_genuine=is_genuine,
        )

        total_time = (time.time() - t_start) * 1000
        logger.info(
            "verify_result job_id=%s",
            job_id,
            extra=_log_extra(
                job_id,
                status=decision_status,
                similarity=decision_similarity,
                liveness_passed=liveness_passed,
                replay_detected=replay_detected,
                total_ms=total_time,
            ),
        )
        logger.warning(
            "stage_times job_id=%s detect_ms=%.3f embed_ms=%.3f search_ms=%.3f total_ms=%.3f faiss_enabled=%s",
            job_id,
            float(pipeline_time.get("detect_ms", 0.0)),
            float(pipeline_time.get("encode_ms", 0.0)),
            float(search_time),
            float(total_time),
            bool(settings.FAISS_ENABLED),
        )
        print(
            f"stage_times job_id={job_id} "
            f"detect_ms={float(pipeline_time.get('detect_ms', 0.0)):.3f} "
            f"embed_ms={float(pipeline_time.get('encode_ms', 0.0)):.3f} "
            f"search_ms={float(search_time):.3f} "
            f"total_ms={float(total_time):.3f} "
            f"faiss_enabled={bool(settings.FAISS_ENABLED)}",
            flush=True,
        )
        _record_verify_result(decision_status)
        _record_liveness_result(liveness_passed)
        _observe_verify_latency(t_start)

        return {
            "status": decision_status,
            "user_id": decision_user_id if decision_status == "match" else None,
            "similarity": float(decision_similarity),
            "margin": float(margin),
            "liveness_passed": liveness_passed,
            "replay_detected": replay_detected
        }
