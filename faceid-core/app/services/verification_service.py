# app/services/verification_service.py - Сервис верификации

import logging
from typing import Dict, Any, Optional
import numpy as np
import time

from app.ml.pipeline import FacePipeline
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.verification_repo import VerificationRepository
from app.services.decision_service import DecisionService
from app.services.liveness_service import LivenessService
from app.services.anti_replay_service import AntiReplayService
from app.services.search_service import SearchService
try:
    from app.monitoring.metrics import (
        VERIFY_LATENCY,
        VERIFY_RESULT_COUNTER,
        LIVENESS_RESULT_COUNTER,
        PIPELINE_STAGE_DURATION,
    )
    METRICS_ENABLED = True
except Exception:
    METRICS_ENABLED = False

THRESHOLD = 0.35  # minimal similarity threshold для уверенного совпадения
LOW_CONFIDENCE_THRESHOLD = 0.5  # нижний порог для режима low_confidence
logger = logging.getLogger("verification")


def _metric_bool(value: Optional[bool]) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "none"


def _observe_stage(stage: str, duration_ms: float) -> None:
    if METRICS_ENABLED:
        try:
            PIPELINE_STAGE_DURATION.labels(stage=stage).observe(duration_ms)
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
        embedding_repo: EmbeddingRepository,
        verification_repo: VerificationRepository,
        search_service: SearchService | None = None,
        pipeline: Any | None = None,
    ):
        self.embedding_repo = embedding_repo
        self.verification_repo = verification_repo
        self.pipeline = pipeline if pipeline is not None else FacePipeline()
        self.search_service = (
            search_service if search_service is not None else SearchService(embedding_repo)
        )

    async def verify_face(
        self,
        image_bytes: bytes,
        user_id: Optional[str] = None,
        require_liveness: bool = False,
        check_replay: bool = True
    ) -> Dict[str, Any]:
        return await self._verify_face_impl(
            image_bytes=image_bytes,
            user_id=user_id,
            require_liveness=require_liveness,
            check_replay=check_replay,
        )

    async def _verify_face_impl(
        self,
        image_bytes: bytes,
        user_id: Optional[str] = None,
        require_liveness: bool = False,
        check_replay: bool = True
    ) -> Dict[str, Any]:

        t_start = time.time()

        def record_result(status: str, liveness_passed: Optional[bool]) -> None:
            return

        # Convert user_id to int or None
        try:
            user_id_int: Optional[int] = int(user_id) if user_id else None
        except (ValueError, TypeError):
            user_id_int = None

        # ANTI-REPLAY CHECK (just a signal, don't block)
        replay_detected = False
        if check_replay:
            replay_detected = not AntiReplayService.check(image_bytes)

        logger.info(
            "verify_started",
            extra={
                "user_id": user_id,
                "require_liveness": require_liveness,
            }
        )

        try:
            result = self.pipeline.process(image_bytes)
        except Exception as e:
            logger.exception("pipeline_failed", extra={"error": str(e)})

            await self.verification_repo.create_log(
                user_id=user_id_int,
                similarity=0.0,
                success=False,
                margin=None,
                liveness_score=None,
                is_genuine=None
            )
            record_result("processing_failed", False)
            _record_verify_result("processing_failed")
            _observe_verify_latency(t_start)

            return {
                "status": "processing_failed",
                "liveness_passed": False,
                "replay_detected": replay_detected
            }

        embedding: np.ndarray = result["embedding"]
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0.0:
            record_result("no_match", None)
            _record_verify_result("no_match")
            _observe_verify_latency(t_start)
            return {
                "status": "no_match",
                "similarity": 0.0,
                "liveness_passed": None,
                "replay_detected": replay_detected
            }
        embedding = embedding / norm
        liveness_signals: Dict[str, Any] = result.get("liveness", {})
        pipeline_time = result.get("timings", {})

        if "total_pipeline_ms" in pipeline_time:
            _observe_stage("pipeline", float(pipeline_time["total_pipeline_ms"]))
        if "detect_ms" in pipeline_time:
            _observe_stage("detect", float(pipeline_time["detect_ms"]))
        if "encode_ms" in pipeline_time:
            _observe_stage("embed", float(pipeline_time["encode_ms"]))
        if "liveness_ms" in pipeline_time:
            _observe_stage("liveness", float(pipeline_time["liveness_ms"]))

        logger.info(
            "pipeline_completed",
            extra={
                "timings": pipeline_time
            }
        )

        # Compute composite liveness score
        liveness_result = LivenessService.fuse(liveness_signals)
        liveness_score = liveness_result["score"]
        liveness_risk = liveness_result["risk"]

        # LIVENESS CHECK
        liveness_passed = LivenessService.is_passed(liveness_signals)
        if require_liveness and not liveness_passed:

            await self.verification_repo.create_log(
                user_id=user_id_int,
                similarity=0.0,
                success=False,
                margin=None,
                liveness_score=liveness_score,
                is_genuine=None
            )
            record_result("spoof_detected", False)
            _record_verify_result("spoof_detected")
            _record_liveness_result(False)
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
        top_k = await self.search_service.search_top_k(embedding, k=2)
        search_time = (time.time() - t0) * 1000
        _observe_stage("search", search_time)

        top1_similarity = top_k[0]["similarity"] if top_k else 0.0
        top2_similarity = top_k[1]["similarity"] if len(top_k) > 1 else 0.0
        margin = top1_similarity - top2_similarity
        logger.info(
            "search_completed",
            extra={
                "top1_similarity": top1_similarity,
                "top2_similarity": top2_similarity,
                "margin": margin,
                "search_ms": search_time
            }
        )

        if not top_k:
            total_time = (time.time() - t_start) * 1000
            logger.info(
                "verify_result",
                extra={
                    "status": "no_match",
                    "similarity": 0.0,
                    "liveness_passed": liveness_passed,
                    "replay_detected": replay_detected,
                    "total_ms": total_time
                }
            )
            record_result("no_match", liveness_passed)
            return {
                "status": "no_match",
                "similarity": 0.0,
                "margin": 0.0,
                "timings": {
                    "pipeline": pipeline_time,
                    "search_ms": search_time,
                    "total_ms": total_time
                },
                "replay_detected": replay_detected
            }

        similarity = top1_similarity
        matched_user_id = top_k[0]["user_id"]

        # Check if matches requested user_id (by external_id)
        is_genuine: bool | None = None
        if user_id:
            from app.db.repositories.user_repo import UserRepository
            user_repo = UserRepository(self.embedding_repo.db)
            expected_user = await user_repo.get_by_external_id(user_id)

            if expected_user:
                expected_user_id = getattr(expected_user, 'id')

                # Get all embeddings for this user and find best score
                embeddings = await self.search_service.search_user_embeddings(expected_user_id)
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
                    is_genuine = False
                    similarity = 0.0

                if not is_genuine:
                    await self.verification_repo.create_log(
                        user_id=expected_user_id,
                        similarity=similarity,
                        success=False,
                        margin=margin,
                        liveness_score=liveness_score,
                        is_genuine=is_genuine
                    )
                    record_result("no_match", liveness_passed)
                    return {
                        "status": "no_match",
                        "liveness_passed": liveness_passed,
                        "liveness": {
                            "score": liveness_score,
                            "risk": liveness_risk
                        },
                        "replay_detected": replay_detected
                    }

        # Decision logic
        best_score = similarity
        # binary decision for match
        is_match = similarity >= THRESHOLD

        # extended status
        if is_match:
            status = "match"
            confidence = "high"
        else:
            confidence_score = (
                similarity * 0.7
                + margin * 0.2
                + liveness_score * 0.1
            )

            if confidence_score >= 0.6:
                status = "low_confidence"
                confidence = "medium"
            else:
                status = "no_match"
                confidence = "low"

        await self.verification_repo.create_log(
            user_id=top_k[0]["user_id"] if top_k else None,
            similarity=similarity,
            margin=margin,
            liveness_score=liveness_score,
            success=(status == "match")
        )

        total_time = (time.time() - t_start) * 1000
        logger.info(
            "verify_result",
            extra={
                "status": status,
                "similarity": best_score,
                "liveness_passed": liveness_passed,
                "replay_detected": replay_detected,
                "total_ms": total_time
            }
        )
        record_result(status, liveness_passed)
        _record_verify_result(status)
        _record_liveness_result(liveness_passed)
        _observe_verify_latency(t_start)

        return {
            "status": status,
            "user_id": matched_user_id if status == "match" else None,
            "similarity": float(best_score),
            "margin": float(margin),
            "liveness_passed": liveness_passed,
            "replay_detected": replay_detected
        }
