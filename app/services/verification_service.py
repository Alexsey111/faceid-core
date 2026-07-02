# app/services/verification_service.py - Сервис верификации

import asyncio
import logging
from typing import Dict, Any, Optional
import numpy as np
import time

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
        QUALITY_REJECT_COUNTER,
        QUALITY_GATE_PRE_MS,
        QUALITY_GATE_FACE_MS,
        PREPROCESS_MS,
        ALIGN_CROP_MS,
        LIVENESS_MS,
        PIPELINE_STAGE_DURATION,
        PIPELINE_MS,
        DETECT_MS,
        ENCODE_MS,
        VECTOR_SEARCH_MS,
        RESULT_WRITE_MS,
    )
    METRICS_ENABLED = True
except Exception:
    IS_GENUINE_MODE = None
    METRICS_ENABLED = False

logger = logging.getLogger("verification")

# Модульный синглтон-семафор, ограничивающий число параллельных ML-инференсов
# в event loop API-процесса. get_verification_service(db) создаёт новый инстанс
# сервиса на каждый запрос, поэтому семафор обязан быть общим (модульным).
# Создаётся лениво: в однопоточном event loop проверка + создание без await
# между ними атомарны, отдельный lock не нужен. По умолчанию совпадает с
# доказанной конкуренцией fast_worker (FAST_WORKER_MAX_CONCURRENCY=4).
_infer_semaphore: Optional[asyncio.Semaphore] = None


def _get_infer_semaphore() -> asyncio.Semaphore:
    global _infer_semaphore
    if _infer_semaphore is None:
        _infer_semaphore = asyncio.Semaphore(settings.API_INFER_CONCURRENCY)
    return _infer_semaphore


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
            PIPELINE_STAGE_DURATION.labels(stage=stage).observe(duration_ms / 1000.0)
        except Exception:
            pass


def _observe_pipeline_metrics(pipeline_time: dict[str, Any]) -> None:
    if not METRICS_ENABLED:
        return

    try:
        if "total_pipeline_ms" in pipeline_time:
            PIPELINE_MS.observe(float(pipeline_time["total_pipeline_ms"]))
        if "preprocess_ms" in pipeline_time:
            PREPROCESS_MS.observe(float(pipeline_time["preprocess_ms"]))
        if "align_crop_ms" in pipeline_time:
            ALIGN_CROP_MS.observe(float(pipeline_time["align_crop_ms"]))
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
            latency_seconds = time.time() - start_time
            VERIFY_LATENCY.observe(latency_seconds)
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


def _record_liveness_fail() -> None:
    if METRICS_ENABLED:
        try:
            LIVENESS_FAIL_COUNT.inc()
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

        if result.get("status") == "quality_reject":
            return {
                "status": "quality_reject",
                "reason": result.get("quality_reason"),
                "quality_details": result.get("quality_details", {}),
                "timings": result.get("timings", {}),
                "bbox": result.get("bbox"),
                "bbox_source": result.get("bbox_source"),
            }

        if result.get("status") == "retry":
            # Окклюзия (маска/очки): не исход верификации, а запрос пере-съёмки.
            # reason="remove_occlusion"; occlusion_flags лежат в quality_details.
            return {
                "status": "retry",
                "reason": result.get("quality_reason") or "remove_occlusion",
                "quality_details": result.get("quality_details", {}),
                "timings": result.get("timings", {}),
                "bbox": result.get("bbox"),
                "bbox_source": result.get("bbox_source"),
            }

        if result.get("status") == "spoof":
            liveness_score = result.get("liveness_score", result.get("confidence"))
            _record_liveness_fail()
            return {
                "status": "spoof",
                "liveness_passed": False,
                "liveness": {
                    "score": liveness_score,
                    "risk": "spoof",
                },
                "bbox": result.get("bbox"),
                "bbox_source": result.get("bbox_source"),
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
            "liveness_passed": result.get("liveness_passed"),
            "liveness_score": result.get("liveness_score"),
            "bbox": result.get("bbox"),
            "bbox_source": result.get("bbox_source"),
            "quality_details": result.get("quality_details"),
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

    async def verify_from_pipeline_result(
        self,
        features: Dict[str, Any],
        image_bytes: bytes,
        user_id: Optional[str] = None,
        require_liveness: bool = False,
        check_replay: bool = True,
        job_id: Optional[str] = None,
        t_start: float | None = None,
        top_k: list[dict[str, Any]] | None = None,
        image_hash: str | None = None,
    ) -> Dict[str, Any]:
        if self.embedding_repo is None:
            raise RuntimeError("EmbeddingRepository is required for verify_face")

        t_start = time.time() if t_start is None else t_start
        pipeline_time = dict(features.get("timings", {}))
        features["timings"] = pipeline_time
        service_timings: dict[str, float] = {}

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
        anti_replay_ms = 0.0
        if check_replay:
            t_replay = time.perf_counter()
            try:
                if image_hash:
                    replay_detected = not AntiReplayService.check_with_hash(image_hash)
                else:
                    replay_detected = not AntiReplayService.check(image_bytes)
            except Exception as exc:
                logger.warning(
                    "anti_replay_unavailable job_id=%s error=%s",
                    job_id,
                    exc,
                )
                replay_detected = False
            finally:
                anti_replay_ms = (time.perf_counter() - t_replay) * 1000.0
                service_timings["anti_replay_ms"] = anti_replay_ms
                pipeline_time["anti_replay_ms"] = anti_replay_ms
                if job_id is not None:
                    _observe_stage("anti_replay", anti_replay_ms)

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
                commit=False,
            )

        async def _store_verification_log_timed(
            *,
            user_id: int | None,
            similarity: float,
            success: bool,
            margin: float | None = None,
            liveness_score: float | None = None,
            is_genuine: bool | None = None,
            query_embedding: np.ndarray | None = None,
            top1_user_id: int | None = None,
        ) -> float:
            t_write = time.perf_counter()
            await _store_verification_log(
                user_id=user_id,
                similarity=similarity,
                success=success,
                margin=margin,
                liveness_score=liveness_score,
                is_genuine=is_genuine,
                query_embedding=query_embedding,
                top1_user_id=top1_user_id,
            )
            verification_log_write_ms = (time.perf_counter() - t_write) * 1000.0
            pipeline_time["verification_log_write_ms"] = verification_log_write_ms
            if job_id is not None:
                _observe_stage("verification_log_write", verification_log_write_ms)
            return verification_log_write_ms

        logger.info(
            "verify_started job_id=%s",
            job_id,
            extra=_log_extra(
                job_id,
                user_id=user_id,
                require_liveness=require_liveness,
            ),
        )

        if features.get("status") == "retry":
            # Окклюзия (маска/очки) → status="retry", reason="remove_occlusion".
            # Это НЕ исход верификации (не match/no_match), поэтому verification_log
            # не пишем (не должно искажать FRR/TAR). Метрику reject-счётчика инкрементим
            # отдельно для наблюдаемости (сколько попыток ушло в «снимите окклюзию»).
            pipeline_time = features.get("timings", {})
            reason = features.get("reason") or "remove_occlusion"

            if job_id is None:
                _observe_pipeline_metrics(pipeline_time)
                if "quality_gate_face_ms" in pipeline_time:
                    _observe_stage("quality_gate_face", float(pipeline_time["quality_gate_face_ms"]))

            if METRICS_ENABLED:
                try:
                    QUALITY_REJECT_COUNTER.labels(reason=reason).inc()
                    if "quality_gate_face_ms" in pipeline_time:
                        QUALITY_GATE_FACE_MS.observe(float(pipeline_time["quality_gate_face_ms"]))
                except Exception:
                    pass
            _record_verify_result(reason)
            _observe_verify_latency(t_start)

            return {
                "status": "retry",
                "reason": reason,
                "quality_details": features.get("quality_details", {}),
                "error_code": reason,
                "liveness_passed": None,
                "replay_detected": replay_detected,
                "bbox": features.get("bbox"),
                "bbox_source": features.get("bbox_source"),
                "timings": pipeline_time,
                "service_timings": service_timings,
            }

        if features.get("status") == "quality_reject":
            pipeline_time = features.get("timings", {})
            reason = features.get("reason") or "quality_reject"

            if job_id is None:
                _observe_pipeline_metrics(pipeline_time)
                if "total_pipeline_ms" in pipeline_time:
                    _observe_stage("pipeline", float(pipeline_time["total_pipeline_ms"]))
                if "quality_gate_pre_ms" in pipeline_time:
                    _observe_stage("quality_gate_pre", float(pipeline_time["quality_gate_pre_ms"]))
                if "quality_gate_face_ms" in pipeline_time:
                    _observe_stage("quality_gate_face", float(pipeline_time["quality_gate_face_ms"]))

            await _store_verification_log_timed(
                user_id=user_id_int,
                similarity=0.0,
                success=False,
                margin=None,
                liveness_score=None,
                is_genuine=None,
            )
            if METRICS_ENABLED:
                try:
                    QUALITY_REJECT_COUNTER.labels(reason=reason).inc()
                    if "quality_gate_pre_ms" in pipeline_time:
                        QUALITY_GATE_PRE_MS.observe(
                            float(pipeline_time["quality_gate_pre_ms"])
                        )

                    if "quality_gate_face_ms" in pipeline_time:
                        QUALITY_GATE_FACE_MS.observe(
                            float(pipeline_time["quality_gate_face_ms"])
                        )
                except Exception:
                    pass
            _record_verify_result(reason or "quality_reject")
            _record_liveness_result(None)
            _observe_verify_latency(t_start)

            return {
                "status": "quality_reject",
                "reason": reason,
                "quality_details": features.get("quality_details", {}),
                "error_code": reason or "quality_reject",
                "liveness_passed": None,
                "replay_detected": replay_detected,
                "bbox": features.get("bbox"),
                "bbox_source": features.get("bbox_source"),
                "timings": pipeline_time,
                "service_timings": service_timings,
            }

        if features.get("status") == "spoof":
            liveness_score = float(features.get("liveness_score", 0.0) or 0.0)
            liveness_risk = "spoof"

            await _store_verification_log_timed(
                user_id=user_id_int,
                similarity=0.0,
                success=False,
                margin=None,
                liveness_score=liveness_score,
                is_genuine=None,
            )
            _record_verify_result("spoof_detected")
            _record_liveness_result(False)
            _record_liveness_fail()
            _observe_verify_latency(t_start)

            return {
                "status": "spoof_detected",
                "liveness_passed": False,
                "liveness": {
                    "score": liveness_score,
                    "risk": liveness_risk,
                },
                "replay_detected": replay_detected,
                "bbox": features.get("bbox"),
                "bbox_source": features.get("bbox_source"),
                "timings": pipeline_time,
                "service_timings": service_timings,
            }

        embedding = features["embedding"]

        if job_id is None:
            _observe_pipeline_metrics(pipeline_time)

            if "total_pipeline_ms" in pipeline_time:
                _observe_stage("pipeline", float(pipeline_time["total_pipeline_ms"]))
            if "preprocess_ms" in pipeline_time:
                _observe_stage("preprocess", float(pipeline_time["preprocess_ms"]))
            if "detect_ms" in pipeline_time:
                _observe_stage("detect", float(pipeline_time["detect_ms"]))
            if "align_crop_ms" in pipeline_time:
                _observe_stage("align_crop", float(pipeline_time["align_crop_ms"]))
            if "encode_ms" in pipeline_time:
                _observe_stage("encode", float(pipeline_time["encode_ms"]))
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

            await _store_verification_log_timed(
                user_id=user_id_int,
                similarity=0.0,
                success=False,
                margin=None,
                liveness_score=liveness_score,
                is_genuine=None,
            )
            _record_verify_result("spoof_detected")
            _record_liveness_result(False)
            _record_liveness_fail()
            _observe_verify_latency(t_start)

            return {
                "status": "spoof_detected",
                "liveness_passed": False,
                "liveness": {
                    "score": liveness_score,
                    "risk": liveness_risk
                },
                "replay_detected": replay_detected,
                "bbox": features.get("bbox"),
                "bbox_source": features.get("bbox_source"),
                "timings": pipeline_time,
            }

        # VECTOR SEARCH (top-2 for margin calculation)
        if top_k is None:
            t0 = time.time()
            top_k = await search_service.search_top_k(embedding, k=2)
            search_time = (time.time() - t0) * 1000
            service_timings["search_ms"] = search_time
            pipeline_time["vector_search_ms"] = search_time
            if job_id is not None:
                _observe_stage("vector_search", search_time)
                if METRICS_ENABLED:
                    try:
                        VECTOR_SEARCH_MS.observe(search_time)
                    except Exception:
                        pass
            _observe_stage("search", search_time)
        else:
            search_time = 0.0
            service_timings["search_ms"] = 0.0

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
            t_is_genuine = time.perf_counter()
            is_genuine = await _resolve_is_genuine(
                embedding_repo.db,
                expected_external_user_id=user_id,
                query_embedding=embedding,
                top1_user_id=matched_user_id,
            )
            is_genuine_ms = (time.perf_counter() - t_is_genuine) * 1000.0
            pipeline_time["is_genuine_ms"] = is_genuine_ms
            if job_id is not None:
                _observe_stage("is_genuine", is_genuine_ms)

        t_decision = time.perf_counter()
        decision = self.make_decision(embedding, top_k, liveness_signals, user_id=user_id)
        decision_ms = (time.perf_counter() - t_decision) * 1000.0
        service_timings["decision_ms"] = decision_ms
        pipeline_time["decision_ms"] = decision_ms
        if job_id is not None:
            _observe_stage("decision", decision_ms)
        decision_status = decision["status"]
        decision_similarity = float(decision.get("similarity", similarity))
        decision_user_id = decision.get("user_id", matched_user_id)

        await _store_verification_log_timed(
            user_id=decision_user_id,
            similarity=decision_similarity,
            margin=margin,
            liveness_score=liveness_score,
            success=(decision_status == "match"),
            is_genuine=is_genuine,
            query_embedding=embedding,
            top1_user_id=matched_user_id,
        )
        service_timings["verification_log_write_ms"] = float(
            pipeline_time.get("verification_log_write_ms", 0.0)
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
        detect_ms = float(pipeline_time.get("detect_ms", 0.0))
        if detect_ms == 0.0:
            detect_ms = float(pipeline_time.get("fast_detect_ms", 0.0)) + float(
                pipeline_time.get("fallback_detect_ms", 0.0)
            )

        logger.warning(
            "stage_times job_id=%s detect_ms=%.3f embed_ms=%.3f search_ms=%.3f total_ms=%.3f faiss_enabled=%s",
            job_id,
            detect_ms,
            float(pipeline_time.get("encode_ms", 0.0)),
            float(search_time),
            float(total_time),
            bool(settings.FAISS_ENABLED),
        )
        _record_verify_result(decision_status)
        _record_liveness_result(liveness_passed)
        _observe_verify_latency(t_start)

        logger.warning(
            "verify_service_times job_id=%s anti_replay_ms=%.3f is_genuine_ms=%.3f decision_ms=%.3f verification_log_write_ms=%.3f search_ms=%.3f",
            job_id,
            float(pipeline_time.get("anti_replay_ms", 0.0)),
            float(pipeline_time.get("is_genuine_ms", 0.0)),
            float(pipeline_time.get("decision_ms", 0.0)),
            float(pipeline_time.get("verification_log_write_ms", 0.0)),
            float(pipeline_time.get("vector_search_ms", 0.0)),
        )

        # «Серая» зона margin: match с низким отрывом от 2-го кандидата → сомнение,
        # клиенту рекомендуем active-challenge (turn/nod) через WS-стрим.
        challenge_recommended = (
            decision_status == "match"
            and float(settings.CHALLENGE_MARGIN_LOW) < float(margin) < float(settings.CHALLENGE_MARGIN_HIGH)
        )

        return {
            "status": decision_status,
            "user_id": decision_user_id if decision_status == "match" else None,
            "similarity": float(decision_similarity),
            "margin": float(margin),
            "liveness_passed": liveness_passed,
            "replay_detected": replay_detected,
            "challenge_recommended": bool(challenge_recommended),
            "bbox": features.get("bbox"),
            "bbox_source": features.get("bbox_source"),
            "timings": pipeline_time,
            "service_timings": service_timings,
        }

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
            async with _get_infer_semaphore():
                features = await asyncio.to_thread(self.extract_features, image_bytes)
        except ValueError as e:
            error_message = str(e)

            if "Invalid embedding vector" in error_message:
                error_code = "embedding_error"
                result_status = "no_match"

                _record_verify_result("no_match")
            else:
                error_code = "invalid_image"
                result_status = "processing_failed"

                _record_verify_result("invalid_image")

            _observe_verify_latency(t_start)

            return {
                "status": result_status,
                "error_code": error_code,
                "similarity": 0.0,
                "liveness_passed": None,
                "replay_detected": False,
            }
        except Exception as e:
            logger.exception(
                "pipeline_failed job_id=%s",
                job_id,
                extra=_log_extra(job_id, error=str(e)),
            )

            error_str = str(e).lower()

            if "no face" in error_str:
                error_code = "no_face"
            elif "multiple" in error_str:
                error_code = "multiple_faces"
            elif "decode" in error_str or "image" in error_str:
                error_code = "invalid_image"
            else:
                error_code = "pipeline_error"

            _record_verify_result(error_code)
            _observe_verify_latency(t_start)

            return {
                "status": "processing_failed",
                "error_code": error_code,
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
