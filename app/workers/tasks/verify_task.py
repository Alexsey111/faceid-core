# workers/tasks/verify_task.py - Verification async job task

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import threading

import cv2
import numpy as np

from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.verification_job_repo import VerificationJobRepository
from app.db.repositories.verification_repo import VerificationRepository
from app.db.session import AsyncSessionLocal
from app.core.config import settings
from app.infrastructure.minio_client import MinioClient
from app.infrastructure.redis_client import redis_client
from app.models.verification_job import JobStatus
from app.services.backpressure import decrement_active
from app.services.liveness_service import LivenessService
from app.services.search_service import SearchService
from app.services.verification_service import VerificationService
from app.services.verification_service_factory import get_pipeline
from app.services.webhook_service import notify_direct as _webhook_notify_direct
from app.services.backpressure import QUEUE_DELAY_MS_KEY
try:
    from app.monitoring.metrics import (
        QUEUE_DELAY_MS,
        JOB_AGE_MS,
        PIPELINE_MS,
        PIPELINE_STAGE_DURATION,
        PREPROCESS_MS,
        ALIGN_CROP_MS,
        DETECT_MS,
        ENCODE_MS,
        VECTOR_SEARCH_MS,
        RESULT_WRITE_MS,
        LIVENESS_MS,
        LIVENESS_FAIL_COUNT,
        MINIO_DELETE_FAIL_TOTAL,
    )
    METRICS_ENABLED = True
except Exception:
    METRICS_ENABLED = False
    # Заглушки, чтобы код удаления фото не падал NameError при отсутствии метрик.
    class _NullMetric:
        def labels(self, **kwargs):
            return self
        def inc(self, amount: int = 1):
            pass
    MINIO_DELETE_FAIL_TOTAL = _NullMetric()
from celery.signals import worker_process_init
from app.workers.celery_app import celery_app as app

logger = logging.getLogger(__name__)
worker_logger = logging.getLogger("worker")
JOB_RESULT_TTL = 300
_pipeline_warmed_up: bool = False
_thread_local = threading.local()


def _make_dummy_image_bytes() -> bytes:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Failed to encode warmup image")
    return buffer.tobytes()


def warmup_pipeline() -> None:
    global _pipeline_warmed_up
    if _pipeline_warmed_up:
        return

    try:
        # TEMP: only initialize the pipeline here.
        # Running a dummy image through detection is noisy because it has no face.
        pipeline = get_pipeline()
        if hasattr(pipeline, "_init"):
            pipeline._init()
        _pipeline_warmed_up = True
        logger.info("pipeline_warmup_completed")
    except Exception as exc:
        logger.warning("pipeline_warmup_failed: %s", exc)


def _get_loop() -> asyncio.AbstractEventLoop:
    if not hasattr(_thread_local, "loop") or _thread_local.loop.is_closed():
        _thread_local.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_thread_local.loop)
    return _thread_local.loop


def run_worker_coroutine(coro):
    loop = _get_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@worker_process_init.connect
def _on_worker_process_init(**_kwargs) -> None:
    warmup_pipeline()


def _cache_job_result(job_id: str, payload: dict) -> None:
    try:
        redis_client.setex(f"job:{job_id}", json.dumps(payload), ttl=JOB_RESULT_TTL)
    except Exception:
        pass


def _store_queue_delay_ms(queue_delay_ms: float) -> None:
    try:
        redis_client.set(QUEUE_DELAY_MS_KEY, str(float(queue_delay_ms)), ttl=JOB_RESULT_TTL)
    except Exception:
        pass


def _observe_worker_stage(stage: str, duration_ms: float) -> None:
    if not METRICS_ENABLED or duration_ms <= 0:
        return

    try:
        # PIPELINE_STAGE_DURATION хранится в секундах
        PIPELINE_STAGE_DURATION.labels(stage=stage).observe(float(duration_ms) / 1000.0)
    except Exception:
        pass


def _observe_worker_pipeline_metrics(timings: dict) -> None:
    if not METRICS_ENABLED:
        return

    try:
        if "total_pipeline_ms" in timings:
            PIPELINE_MS.observe(float(timings["total_pipeline_ms"]))
        if "preprocess_ms" in timings:
            PREPROCESS_MS.observe(float(timings["preprocess_ms"]))
            _observe_worker_stage("preprocess", float(timings["preprocess_ms"]))
        if "align_crop_ms" in timings:
            ALIGN_CROP_MS.observe(float(timings["align_crop_ms"]))
            _observe_worker_stage("align_crop", float(timings["align_crop_ms"]))
        detect_ms = 0.0
        if "detect_ms" in timings:
            detect_ms = float(timings["detect_ms"])
        else:
            detect_ms += float(timings.get("fast_detect_ms", 0.0))
            detect_ms += float(timings.get("fallback_detect_ms", 0.0))
        if detect_ms > 0.0:
            DETECT_MS.observe(detect_ms)
            _observe_worker_stage("detect", detect_ms)
        if "encode_ms" in timings:
            encode_ms = float(timings["encode_ms"])
            ENCODE_MS.observe(encode_ms)
            _observe_worker_stage("encode", encode_ms)
        if "liveness_ms" in timings:
            LIVENESS_MS.observe(float(timings["liveness_ms"]))
    except Exception:
        pass


def _observe_result_write(duration_ms: float) -> None:
    if not METRICS_ENABLED:
        return

    try:
        RESULT_WRITE_MS.observe(float(duration_ms))
        _observe_worker_stage("result_write", float(duration_ms))
    except Exception:
        pass


async def _mark_verify_job_failed(job_id: str, image_url: str, error: str) -> None:
    async with AsyncSessionLocal() as db:
        job_repo = VerificationJobRepository(db)
        try:
            await job_repo.update(
                job_id,
                status=JobStatus.failed,
                error=error,
            )
            await db.commit()
        except LookupError:
            logger.error("verify job not found while marking failed job_id=%s", job_id)
            return

    _cache_job_result(
        job_id,
        {
            "job_id": job_id,
            "status": "failed",
            "result": None,
            "error": error,
            "ready": True,
        },
    )

    # Webhook-уведомление о неудаче (ТЗ 3.2). Прямая доставка (Celery-loop
    # короткоживущий): fire-and-forget через очередь не успел бы выполниться.
    try:
        await _webhook_notify_direct(
            job_id, "failed",
            {"job_id": job_id, "status": "failed", "error": error},
        )
    except Exception:
        logger.warning("webhook_dispatch_failed job_id=%s", job_id, exc_info=True)

    try:
        MinioClient().delete_image(image_url)
    except Exception:
        MINIO_DELETE_FAIL_TOTAL.labels(stage="verify_task_failed").inc()
        logger.warning(
            "minio_delete_failed job_id=%s image_url=%s: %s",
            job_id, image_url, exc_info=True,
        )


async def _process_verify_job(
    job_id: str,
    image_url: str,
    user_id: str | None,
    require_liveness: bool,
    request_received_time: float | None,
) -> dict[str, object] | None:
    logger.warning("TASK START job_id=%s", job_id)
    worker_logger.info("worker_job_started", extra={"job_id": job_id})
    print(f"[START] pid={os.getpid()} time={time.time()}", flush=True)
    try:
        minio_client = MinioClient()
        start_total = time.time()
        queue_delay_ms = 0.0
        if request_received_time is not None:
            queue_delay_ms = max(0.0, (start_total - request_received_time) * 1000)
        _store_queue_delay_ms(queue_delay_ms)
        if METRICS_ENABLED:
            try:
                QUEUE_DELAY_MS.observe(float(queue_delay_ms))
            except Exception:
                pass

        # 1. get job and move to processing
        async with AsyncSessionLocal() as db:
            job_repo = VerificationJobRepository(db)

            job = await job_repo.get_by_id(job_id)
            if job is None:
                raise LookupError("Job not found")

            queue_wait_ms = 0.0
            created_at = getattr(job, "created_at", None)
            if created_at is not None:
                queue_wait_ms = max(0.0, (time.time() - created_at.timestamp()) * 1000)
                if METRICS_ENABLED:
                    try:
                        JOB_AGE_MS.observe(float(queue_wait_ms))
                        _observe_worker_stage("dequeue_job_age", float(queue_wait_ms))
                    except Exception:
                        pass

            if job.status in (JobStatus.done, JobStatus.failed):
                return

            await job_repo.update(job_id, status=JobStatus.processing, error=None)
            await db.commit()

            try:
                t0 = time.time()
                image_bytes = minio_client.get_image(image_url)
                logger.warning("job_id=%s download=%.3fs", job_id, time.time() - t0)
            except Exception as exc:
                logger.exception("download failed job_id=%s", job_id)
                raise ValueError("Invalid image payload") from exc

        # 2. ML outside DB
        t0 = time.time()
        service = VerificationService(
            embedding_repo=None,
            verification_repo=None,
            pipeline=get_pipeline(),
        )
        logger.warning("job_id=%s service_init=%.3fs", job_id, time.time() - t0)

        features = service.extract_features(image_bytes)
        logger.warning("job_id=%s ml=%.3fs", job_id, time.time() - t0)
        _observe_worker_pipeline_metrics(features["timings"])
        logger.info(
            "liveness_ms=%.3f encode_ms=%.3f bbox_source=%s",
            float(features["timings"].get("liveness_ms", -1)),
            float(features["timings"].get("encode_ms", -1)),
            features.get("bbox_source"),
        )

        if features.get("status") == "spoof":
            liveness = features["liveness"]
            liveness_score = float(liveness.get("score", 0.0))
            liveness_risk = liveness.get("risk", "spoof")
            if METRICS_ENABLED:
                try:
                    LIVENESS_FAIL_COUNT.inc()
                except Exception:
                    pass

            result = {
                "status": "spoof",
                "liveness_passed": False,
                "liveness": {
                    "score": liveness_score,
                    "risk": liveness_risk,
                },
                "replay_detected": False,
            }

            async with AsyncSessionLocal() as db:
                verification_repo = VerificationRepository(db)
                await verification_repo.create_log(
                    user_id=int(user_id) if user_id else None,
                    similarity=0.0,
                    success=False,
                    margin=None,
                    liveness_score=liveness_score,
                    is_genuine=None,
                    commit=False,
                )

                job_repo = VerificationJobRepository(db)
                await job_repo.update(
                    job_id,
                    status=JobStatus.done,
                    result=result,
                    error=None,
                )

                await db.commit()

            worker_logger.info(
                "worker_job_finished",
                extra={"job_id": job_id, "status": result.get("status")},
            )

            # 152-ФЗ: исходное фото удаляется из MinIO и при spoof-детекции.
            # Раньше spoof-ветка возвращалась ДО delete_image (стр.464 ниже) →
            # фото спуфера оставалось в хранилище навсегда (compliance-баг).
            try:
                minio_client.delete_image(image_url)
            except Exception:
                MINIO_DELETE_FAIL_TOTAL.labels(stage="verify_task_spoof").inc()
                logger.warning(
                    "minio_delete_failed job_id=%s image_url=%s: %s",
                    job_id, image_url, exc_info=True,
                )

            return result

        # optional liveness gate before search/decision
        liveness_passed = LivenessService.is_passed(features["liveness"])
        liveness_result = LivenessService.fuse(features["liveness"])
        liveness_score = liveness_result["score"]
        liveness_risk = liveness_result["risk"]

        log_user_id: int | None = None
        log_similarity = 0.0
        log_success = False
        log_margin: float | None = None
        log_is_genuine: bool | None = None
        # search_time определяется только в else-ветке (search). В
        # liveness-gate-fail-ветке (require_liveness + не passed) поиска нет,
        # но logger.warning ниже использует float(search_time) всегда →
        # раньше был UnboundLocalError и worker падал при passive-spoof-fail.
        search_time = 0.0

        if require_liveness and not liveness_passed:
            if METRICS_ENABLED:
                try:
                    LIVENESS_FAIL_COUNT.inc()
                except Exception:
                    pass
            result = {
                "status": "spoof",
                "liveness_passed": False,
                "liveness": {
                    "score": liveness_score,
                    "risk": liveness_risk,
                },
                "replay_detected": False,
            }
        else:
            # 4. DB search
            t0 = time.time()
            async with AsyncSessionLocal() as db:
                search_service = SearchService(EmbeddingRepository(db))
                top_k = await search_service.search_top_k(features["embedding"])
            search_time = (time.time() - t0) * 1000
            logger.warning("job_id=%s search=%.3fs", job_id, search_time / 1000.0)

            # 5. CPU decision
            decision = service.make_decision(
                features["embedding"],
                top_k,
                features["liveness"],
            )

            log_user_id = decision.get("user_id")
            log_similarity = float(decision.get("similarity", 0.0))
            log_success = decision["status"] == "match"
            log_margin = None

            result = {
                "status": decision["status"],
                "user_id": decision.get("user_id"),
                "similarity": log_similarity,
                "liveness_passed": liveness_passed,
                "liveness": {
                    "score": liveness_score,
                    "risk": liveness_risk,
                },
                "replay_detected": False,
            }

        # 6. DB save
        async with AsyncSessionLocal() as db:
            verification_repo = VerificationRepository(db)
            await verification_repo.create_log(
                user_id=log_user_id,
                similarity=log_similarity,
                success=log_success,
                margin=log_margin,
                liveness_score=liveness_score,
                is_genuine=log_is_genuine,
                commit=False,
            )

            job_repo = VerificationJobRepository(db)
            await job_repo.update(
                job_id,
                status=JobStatus.done,
                result=result,
                error=None,
            )
            await db.commit()

        logger.warning("job_id=%s total=%.3fs", job_id, time.time() - start_total)
        logger.warning(
            "stage_times job_id=%s queue_delay_ms=%.3f queue_wait_ms=%.3f preprocess_ms=%.3f detect_ms=%.3f embed_ms=%.3f search_ms=%.3f total_ms=%.3f faiss_enabled=%s",
            job_id,
            float(queue_delay_ms),
            float(queue_wait_ms),
            float(features["timings"].get("preprocess_ms", 0.0)),
            float(features["timings"].get("detect_ms", 0.0)),
            float(features["timings"].get("encode_ms", 0.0)),
            float(search_time),
            (time.time() - start_total) * 1000,
            bool(settings.FAISS_ENABLED),
        )

        _cache_job_result(
            job_id,
            {
                "job_id": job_id,
                "status": "done",
                "result": result,
                "error": None,
                "ready": True,
                "queue_delay_ms": float(queue_delay_ms),
                "queue_wait_ms": float(queue_wait_ms),
            },
        )

        # Webhook-уведомление об успехе (ТЗ 3.2). Прямая доставка (Celery-loop
        # короткоживущий): payload уже без image_b64 (не лежит в result).
        try:
            await _webhook_notify_direct(
                job_id, "success",
                {"job_id": job_id, "status": "done", "result": result},
            )
        except Exception:
            logger.warning("webhook_dispatch_failed job_id=%s", job_id, exc_info=True)

        try:
            minio_client.delete_image(image_url)
        except Exception:
            MINIO_DELETE_FAIL_TOTAL.labels(stage="verify_task_success").inc()
            logger.warning(
                "minio_delete_failed job_id=%s image_url=%s: %s",
                job_id, image_url, exc_info=True,
            )

        worker_logger.info(
            "worker_job_finished",
            extra={"job_id": job_id, "status": result.get("status")},
        )
    except LookupError:
        logger.error("verify job not found", extra={"job_id": job_id})
        return
    finally:
        print(f"[END] pid={os.getpid()} time={time.time()}", flush=True)


@app.task(
    name="app.workers.tasks.verify_job",
    bind=True,
    max_retries=3,
    soft_time_limit=15,
    time_limit=25,
)
def process_verify_job(
    self,
    job_id: str,
    image_url: str,
    user_id: str | None = None,
    require_liveness: bool = False,
    request_received_time: float | None = None,
):
    # Defense-in-depth active-challenge gate: прямой enqueue в очередь (минуя
    # роуты с route-gate) с require_liveness=true при LIVENESS_ACTIVE_REQUIRED=true
    # — это passive-допуск, который ложит physical-spoof (cutout/print). Маркируем
    # job failed БЕЗ retry (легитимные active-запросы приходят с require_liveness=False).
    if require_liveness and settings.LIVENESS_ACTIVE_REQUIRED:
        logger.error(
            "active_liveness_required job_id=%s — passive admission rejected by policy",
            job_id,
        )
        run_worker_coroutine(
            _mark_verify_job_failed(
                job_id,
                image_url,
                "active_liveness_required: use /api/v1/verify_base64 with "
                "liveness_mode=active + liveness_token",
            )
        )
        return
    try:
        run_worker_coroutine(
            _process_verify_job(job_id, image_url, user_id, require_liveness, request_received_time)
        )
    except LookupError:
        return
    except Exception as exc:
        if self.request.retries < self.max_retries:
            logger.warning(
                "retrying job_id=%s attempt=%s/%s",
                job_id,
                self.request.retries + 1,
                self.max_retries,
                exc_info=True,
            )
            raise self.retry(exc=exc, countdown=min(2 ** self.request.retries, 30))

        logger.error("verify job failed permanently job_id=%s", job_id, exc_info=True)
        run_worker_coroutine(_mark_verify_job_failed(job_id, image_url, str(exc)))
        raise
    finally:
        decrement_active()


verify_task = process_verify_job
