# app/services/verification_service.py - Сервис верификации

import asyncio
import logging
from typing import Dict, Any, Optional
import numpy as np
import time

from fastapi.concurrency import run_in_threadpool

from app.ml.pipeline import FacePipeline
from app.ml.pipeline_v2 import FacePipelineV2
from app.core.config import settings
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.verification_repo import VerificationRepository
from app.db.session import AsyncSessionLocal
from app.db.repositories.user_repo import UserRepository
from app.services.liveness_service import LivenessService
from app.services.anti_replay_service import AntiReplayService
from app.services.search_service import SearchService
try:
    from app.monitoring.metrics import (
        IS_GENUINE_MODE,
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
    IS_GENUINE_MODE = None
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


async def _persist_verification_log_background(
    *,
    user_id: int | None,
    similarity: float,
    success: bool,
    margin: float | None = None,
    liveness_score: float | None = None,
    is_genuine: bool | None = None,
    expected_external_user_id: str | None = None,
    query_embedding: np.ndarray | None = None,
    top1_user_id: int | None = None,
) -> None:
    if user_id is None:
        return

    try:
        async with AsyncSessionLocal() as db:
            computed_is_genuine = await _resolve_is_genuine(
                db,
                expected_external_user_id=expected_external_user_id,
                query_embedding=query_embedding,
                top1_user_id=top1_user_id,
                fallback_is_genuine=is_genuine,
            )

            repo = VerificationRepository(db)
            await repo.create_log(
                user_id=user_id,
                similarity=similarity,
                success=success,
                margin=margin,
                liveness_score=liveness_score,
                is_genuine=computed_is_genuine,
            )
    except Exception:
        logger.exception("background_verification_log_failed user_id=%s", user_id)


async def _resolve_is_genuine(
    db: Any,
    *,
    expected_external_user_id: str | None,
    query_embedding: np.ndarray | None,
    top1_user_id: int | None,
    fallback_is_genuine: bool | None = None,
) -> bool | None:
    if not expected_external_user_id:
        return fallback_is_genuine

    user_repo = UserRepository(db)
    expected_user = await user_repo.get_by_external_id(expected_external_user_id)
    if not expected_user:
        return False

    expected_user_id = getattr(expected_user, "id")

    if settings.USE_SIMPLE_IS_GENUINE or query_embedding is None:
        return bool(top1_user_id == expected_user_id)

    embedding_repo = EmbeddingRepository(db)
    vectors = await embedding_repo.get_user_vectors(expected_user_id)
    if not vectors:
        return False

    centroid = np.mean(np.stack(vectors), axis=0)
    centroid_norm = np.linalg.norm(centroid)
    query = np.asarray(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query)

    if centroid_norm == 0.0 or query_norm == 0.0:
        return False

    centroid = centroid / centroid_norm
    query = query / query_norm
    centroid_similarity = float(np.dot(query, centroid))
    return centroid_similarity >= settings.HIGH_THRESHOLD


class VerificationService:

    def __init__(
        self,
        embedding_repo: EmbeddingRepository | None,
        verification_repo: VerificationRepository | None,
        search_service: SearchService | None = None,
        pipeline: Any | None = None,
        load_pipeline: bool = True,
    ):
        self.embedding_repo = embedding_repo
        self.verification_repo = verification_repo
        active_mode = "simple" if settings.USE_SIMPLE_IS_GENUINE else "centroid"
        if METRICS_ENABLED and IS_GENUINE_MODE is not None:
            IS_GENUINE_MODE.labels(mode=active_mode).set(1)
            IS_GENUINE_MODE.labels(mode="centroid" if active_mode == "simple" else "simple").set(0)
        logger.info(
            "is_genuine_mode=%s",
            active_mode,
        )
        if pipeline is not None:
            self.pipeline = pipeline
        elif load_pipeline:
            self.pipeline = FacePipelineV2() if settings.USE_PIPELINE_V2 else FacePipeline()
        else:
            self.pipeline = None
        self.search_service = search_service
        if self.search_service is None and embedding_repo is not None:
            self.search_service = SearchService(embedding_repo)

    def extract_features(self, image_bytes: bytes) -> dict:
        if self.pipeline is None:
            raise RuntimeError("Pipeline is required for extract_features")

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

    async def verify_face_in_worker(
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

    async def verify_from_pipeline_result(
        self,
        features: Dict[str, Any],
        image_bytes: bytes,
        user_id: Optional[str] = None,
        require_liveness: bool = False,
        check_replay: bool = True,
        job_id: Optional[str] = None,
        t_start: float | None = None,
    ) -> Dict[str, Any]:
        if self.embedding_repo is None:
            raise RuntimeError("EmbeddingRepository is required for verify_face")

        t_start = time.time() if t_start is None else t_start

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
        user_id_external_id = user_id

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

        async def _store_verification_log(
            *,
            user_id: int | None,
            similarity: float,
            success: bool,
            margin: float | None = None,
            liveness_score: float | None = None,
            is_genuine: bool | None = None,
            query_embedding: np.ndarray | None = None,
            top1_user_id: int | None = None,
        ) -> None:
            if job_id is None:
                asyncio.create_task(
                    _persist_verification_log_background(
                        user_id=user_id,
                        similarity=similarity,
                        success=success,
                        margin=margin,
                        liveness_score=liveness_score,
                        is_genuine=is_genuine,
                        expected_external_user_id=user_id_external_id,
                        query_embedding=query_embedding,
                        top1_user_id=top1_user_id,
                    )
                )
                return

            await verification_repo.create_log(
                user_id=user_id,
                similarity=similarity,
                success=success,
                margin=margin,
                liveness_score=liveness_score,
                is_genuine=is_genuine,
            )

        logger.info(
            "verify_started job_id=%s",
            job_id,
            extra=_log_extra(
                job_id,
                user_id=user_id,
                require_liveness=require_liveness,
            ),
        )

        if features.get("status") == "spoof":
            liveness_score = float(features.get("liveness_score", 0.0) or 0.0)
            liveness_risk = "spoof"

            await _store_verification_log(
                user_id=user_id_int,
                similarity=0.0,
                success=False,
                margin=None,
                liveness_score=liveness_score,
                is_genuine=None,
            )
            _record_verify_result("spoof_detected")
            _record_liveness_result(False)
            LIVENESS_FAIL_COUNT.inc()
            _observe_verify_latency(t_start)

            return {
                "status": "spoof_detected",
                "liveness_passed": False,
                "liveness": {
                    "score": liveness_score,
                    "risk": liveness_risk,
                },
                "replay_detected": replay_detected,
            }

        embedding = features["embedding"]
        pipeline_time = features.get("timings", {})

        if job_id is not None:
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

        # Pipeline v2 returns liveness_passed / liveness_score directly.
        # Older pipeline variants still provide a full liveness signal dict.
        raw_liveness_signals = features.get("liveness")
        if raw_liveness_signals is None:
            liveness_score = float(features.get("liveness_score", 0.0) or 0.0)
            liveness_passed = features.get("liveness_passed")
            if liveness_passed is None:
                liveness_passed = liveness_score >= settings.LIVENESS_THRESHOLD

            if liveness_score >= 0.8:
                liveness_risk = "low"
            elif liveness_score >= 0.6:
                liveness_risk = "medium"
            else:
                liveness_risk = "high"

            liveness_signals = {
                "passive": liveness_score,
            }
        else:
            liveness_signals = raw_liveness_signals
            # Compute composite liveness score
            liveness_result = LivenessService.fuse(liveness_signals)
            liveness_score = liveness_result["score"]
            liveness_risk = liveness_result["risk"]

            # LIVENESS CHECK
            liveness_passed = LivenessService.is_passed(liveness_signals)
        if require_liveness and not liveness_passed:

            await _store_verification_log(
                user_id=user_id_int,
                similarity=0.0,
                success=False,
                margin=None,
                liveness_score=liveness_score,
                is_genuine=None,
            )
            _record_verify_result("spoof_detected")
            _record_liveness_result(False)
            LIVENESS_FAIL_COUNT.inc()
            _observe_verify_latency(t_start)

            return {
                "status": "spoof_detected",
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
            is_genuine = await _resolve_is_genuine(
                embedding_repo.db,
                expected_external_user_id=user_id,
                query_embedding=embedding,
                top1_user_id=matched_user_id,
            )

        decision = self.make_decision(embedding, top_k, liveness_signals, user_id=user_id)
        decision_status = decision["status"]
        decision_similarity = float(decision.get("similarity", similarity))
        decision_user_id = decision.get("user_id", matched_user_id)

        await _store_verification_log(
            user_id=decision_user_id,
            similarity=decision_similarity,
            margin=margin,
            liveness_score=liveness_score,
            success=(decision_status == "match"),
            is_genuine=is_genuine,
            query_embedding=embedding,
            top1_user_id=matched_user_id,
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

    async def verify_face_sync(self, image_bytes: bytes) -> dict:
        if self.embedding_repo is None:
            raise RuntimeError("EmbeddingRepository is required for verify_face_sync")
        if self.pipeline is None:
            raise RuntimeError("Pipeline is required for verify_face_sync")

        t_start = time.time()

        # Keep the event loop responsive while the CPU-heavy pipeline runs.
        result = await run_in_threadpool(self.pipeline.process, image_bytes)

        return await self.verify_from_pipeline_result(
            result,
            image_bytes=image_bytes,
            t_start=t_start,
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
                    "replay_detected": False,
                }

            logger.exception(
                "pipeline_failed job_id=%s",
                job_id,
                extra=_log_extra(job_id, error=str(e)),
            )

            _record_verify_result("processing_failed")
            _observe_verify_latency(t_start)

            return {
                "status": "processing_failed",
                "liveness_passed": False,
                "replay_detected": False,
            }
        except Exception as e:
            logger.exception(
                "pipeline_failed job_id=%s",
                job_id,
                extra=_log_extra(job_id, error=str(e)),
            )

            _record_verify_result("processing_failed")
            _observe_verify_latency(t_start)

            return {
                "status": "processing_failed",
                "liveness_passed": False,
                "replay_detected": False,
            }

        return await self.verify_from_pipeline_result(
            features,
            image_bytes=image_bytes,
            user_id=user_id,
            require_liveness=require_liveness,
            check_replay=check_replay,
            job_id=job_id,
            t_start=t_start,
        )
