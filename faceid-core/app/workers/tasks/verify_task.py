# workers/tasks/verify_task.py - Verification async job task

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

import cv2
import numpy as np

from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.verification_job_repo import VerificationJobRepository
from app.db.repositories.verification_repo import VerificationRepository
from app.db.session import AsyncSessionLocal
from app.core.config import settings
from app.ml.pipeline import FacePipeline
from app.infrastructure.minio_client import MinioClient
from app.infrastructure.redis_client import redis_client
from app.models.verification_job import JobStatus
from app.services.backpressure import decrement_active
from app.services.liveness_service import LivenessService
from app.services.search_service import SearchService
from app.services.verification_service import VerificationService
from celery.signals import worker_process_init
from app.workers.celery_app import celery_app as app

logger = logging.getLogger(__name__)
JOB_RESULT_TTL = 300
_pipeline: FacePipeline | None = None
_worker_loop: asyncio.AbstractEventLoop | None = None
_pipeline_warmed_up: bool = False


def get_pipeline() -> FacePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = FacePipeline()
    return _pipeline


def get_worker_loop() -> asyncio.AbstractEventLoop:
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
    return _worker_loop


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
        get_pipeline()._init()
        _pipeline_warmed_up = True
        logger.info("pipeline_warmup_completed")
    except Exception as exc:
        logger.warning("pipeline_warmup_failed: %s", exc)


@worker_process_init.connect
def _on_worker_process_init(**_kwargs) -> None:
    warmup_pipeline()


def _cache_job_result(job_id: str, payload: dict) -> None:
    try:
        redis_client.setex(f"job:{job_id}", json.dumps(payload), ttl=JOB_RESULT_TTL)
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

    try:
        MinioClient().delete_image(image_url)
    except Exception:
        pass


async def _process_verify_job(
    job_id: str,
    image_url: str,
    user_id: str | None,
    require_liveness: bool,
    request_received_time: float | None,
) -> None:
    logger.warning("TASK START job_id=%s", job_id)
    print(f"[START] pid={os.getpid()} time={time.time()}", flush=True)
    try:
        minio_client = MinioClient()
        start_total = time.time()
        queue_delay_ms = 0.0
        if request_received_time is not None:
            queue_delay_ms = max(0.0, (start_total - request_received_time) * 1000)

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

        if require_liveness and not liveness_passed:
            result = {
                "status": "spoof_detected",
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

        try:
            minio_client.delete_image(image_url)
        except Exception:
            pass
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
    try:
        loop = get_worker_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
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
        loop = get_worker_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_mark_verify_job_failed(job_id, image_url, str(exc)))
        raise
    finally:
        decrement_active()


verify_task = process_verify_job
