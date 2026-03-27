# app/workers/verify_worker.py

import base64
import asyncio
import json
import redis
import time
from typing import Any, Tuple, cast

import cv2
import numpy as np

from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.verification_repo import VerificationRepository
from app.db.session import AsyncSessionLocal
from app.monitoring.metrics import (
    ASYNC_JOB_COMPLETED_TOTAL,
    ASYNC_JOB_EXPIRED_TOTAL,
    ASYNC_JOB_E2E_LATENCY_MS,
    ASYNC_JOB_PROCESSING_MS,
    ASYNC_JOB_QUEUE_DELAY_MS,
    ASYNC_JOB_TOTAL_LATENCY_MS,
    QUEUE_LENGTH,
    JOB_AGE_MS,
    VERIFY_INFLIGHT_CURRENT,
    VERIFY_REJECTED_JOBS,
    VERIFY_WORKER_UTILIZATION,
)
from prometheus_client import start_http_server
from app.services.search_service import SearchService
from app.services.verify_job_queue import VerifyJobQueue
from app.services.verify_result_store import VerifyResultStore
from app.services.verification_service import VerificationService

redis_client = redis.Redis(host="redis", port=6379, db=0)
QUEUE_NAME = "face_verify_queue"
MAX_QUEUE_WAIT = 2.0
MAX_JOB_AGE_MS = 3000
MAX_IMAGE_SIDE = 640
semaphore = asyncio.Semaphore(2)
_PIPELINE: Any | None = None


def _update_inflight_metrics() -> None:
    inflight_raw = cast(bytes | None, redis_client.get("inflight_jobs"))
    inflight = int(inflight_raw.decode("utf-8")) if inflight_raw else 0
    VERIFY_INFLIGHT_CURRENT.set(inflight)
    VERIFY_WORKER_UTILIZATION.set(
        min(1.0, inflight / float(VerifyJobQueue.INFLIGHT_LIMIT))
    )


def _update_queue_length_metric() -> None:
    queue_length = float(cast(int, redis_client.llen(QUEUE_NAME)))
    QUEUE_LENGTH.set(queue_length)


def _get_batch_size(queue_len: int) -> int:
    if queue_len < 10:
        return 2
    if queue_len < 30:
        return 4
    return 6


def _build_metrics(created_at: float, started_at: float, finished_at: float) -> dict[str, float]:
    return {
        "created_at": created_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "queue_delay": started_at - created_at,
        "processing_time": finished_at - started_at,
        "total_latency": finished_at - created_at,
    }


def _observe_async_job_metrics(metrics: dict[str, float], *, completed: bool = False, expired: bool = False) -> None:
    total_latency_ms = metrics["total_latency"] * 1000.0
    e2e_latency_ms = (metrics["finished_at"] - metrics["created_at"]) * 1000.0
    queue_delay_ms = metrics["queue_delay"] * 1000.0
    processing_ms = metrics["processing_time"] * 1000.0

    ASYNC_JOB_TOTAL_LATENCY_MS.observe(total_latency_ms)
    ASYNC_JOB_E2E_LATENCY_MS.observe(e2e_latency_ms)
    ASYNC_JOB_QUEUE_DELAY_MS.observe(queue_delay_ms)
    ASYNC_JOB_PROCESSING_MS.observe(processing_ms)

    if completed:
        ASYNC_JOB_COMPLETED_TOTAL.inc()
    if expired:
        ASYNC_JOB_EXPIRED_TOTAL.inc()


def _print_metrics(job_id: str, metrics: dict[str, float]) -> None:
    print(
        f"[METRICS] job={job_id} "
        f"queue={metrics['queue_delay']:.2f}s "
        f"proc={metrics['processing_time']:.2f}s "
        f"total={metrics['total_latency']:.2f}s"
    )


def _finish_expired_job(job_id: str, created_at: float, started_at: float) -> None:
    finished_at = time.time()
    metrics = _build_metrics(created_at, started_at, finished_at)
    age_ms = (finished_at - created_at) * 1000.0
    _observe_async_job_metrics(metrics, expired=True)
    JOB_AGE_MS.observe(age_ms)
    _print_metrics(job_id, metrics)
    VERIFY_REJECTED_JOBS.labels(reason="expired").inc()
    VerifyResultStore.set_expired(job_id, metrics)
    redis_client.decr("inflight_jobs")
    _update_inflight_metrics()


def _reject_stale_job(job_id: str, created_at: float, observed_at: float) -> None:
    metrics = _build_metrics(created_at, observed_at, observed_at)
    age_ms = (observed_at - created_at) * 1000.0
    _observe_async_job_metrics(metrics, expired=True)
    JOB_AGE_MS.observe(age_ms)
    _print_metrics(job_id, metrics)
    VERIFY_REJECTED_JOBS.labels(reason="expired").inc()
    VerifyResultStore.set_expired(job_id, metrics)
    redis_client.decr("inflight_jobs")
    _update_inflight_metrics()


def _downscale_image_bytes(image_bytes: bytes, max_side: int = MAX_IMAGE_SIDE) -> bytes:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if decoded is None:
        return image_bytes

    height, width = decoded.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image_bytes

    scale = max_side / float(longest_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(decoded, (new_width, new_height), interpolation=cv2.INTER_AREA)

    success, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        return image_bytes
    return encoded.tobytes()


async def warmup():
    global _PIPELINE

    print("WARMUP: starting pipeline warmup...")

    dummy_image = b"\x00" * 1024  # just noise; pipeline is expected to fail

    async with AsyncSessionLocal() as db:
        service = VerificationService(
            embedding_repo=EmbeddingRepository(db),
            verification_repo=VerificationRepository(db),
            search_service=SearchService(EmbeddingRepository(db)),
            load_pipeline=True,
        )
        pipeline = service.pipeline
        if pipeline is None:
            raise RuntimeError("Pipeline is required for warmup")

        _PIPELINE = pipeline

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                pipeline.process,
                dummy_image,
            )
        except Exception:
            pass

    print("WARMUP: done")


async def collect_batch() -> list[dict[str, Any]]:
    queue_len = cast(int, redis_client.llen(QUEUE_NAME))
    _update_queue_length_metric()

    jobs: list[dict[str, Any]] = []
    batch_size = _get_batch_size(queue_len)

    for _ in range(batch_size):
        raw = cast(bytes | None, redis_client.rpop(QUEUE_NAME))
        if raw is None:
            break

        job = json.loads(raw.decode("utf-8"))
        created_at = job.get("created_at", time.time())
        now = time.time()
        age_ms = (now - created_at) * 1000.0

        if age_ms > MAX_JOB_AGE_MS or (now - created_at) > MAX_QUEUE_WAIT:
            _reject_stale_job(job["job_id"], created_at, now)
            continue

        jobs.append(job)

    _update_queue_length_metric()
    return jobs


async def process_batch(job_datas: list[dict[str, Any]]):
    async with semaphore:
        pipeline = _PIPELINE
        if pipeline is None:
            raise RuntimeError("Pipeline is required for batch processing")

        batch_started_at = time.time()
        prepared_jobs: list[dict[str, Any]] = []

        for job_data in job_datas:
            job_id = job_data["job_id"]
            payload = job_data["payload"]
            created_at = job_data.get("created_at", batch_started_at)
            now = time.time()
            age_ms = (now - created_at) * 1000.0

            if age_ms > MAX_JOB_AGE_MS:
                metrics = _build_metrics(created_at, batch_started_at, now)
                _observe_async_job_metrics(metrics, expired=True)
                JOB_AGE_MS.observe(age_ms)
                _print_metrics(job_id, metrics)
                VERIFY_REJECTED_JOBS.labels(reason="expired").inc()
                VerifyResultStore.set_expired(job_id, metrics)
                redis_client.decr("inflight_jobs")
                _update_inflight_metrics()
                continue

            if now - created_at > MAX_QUEUE_WAIT:
                metrics = _build_metrics(created_at, batch_started_at, now)
                _observe_async_job_metrics(metrics, expired=True)
                JOB_AGE_MS.observe(age_ms)
                _print_metrics(job_id, metrics)
                VerifyResultStore.set_expired(job_id, metrics)
                redis_client.decr("inflight_jobs")
                _update_inflight_metrics()
                continue

            if (batch_started_at - created_at) > MAX_QUEUE_WAIT:
                _finish_expired_job(job_id, created_at, batch_started_at)
                continue

            try:
                image_bytes = base64.b64decode(payload["image_b64"])
                image_bytes = _downscale_image_bytes(image_bytes)
                t0 = time.time()
                prepared = pipeline.prepare_face_input(image_bytes)
                prep_ms = (time.time() - t0) * 1000.0
                print(f"[PREP] {prep_ms:.2f} ms")
                prepared_jobs.append(
                    {
                        "job_id": job_id,
                        "payload": payload,
                        "image_bytes": image_bytes,
                        "created_at": created_at,
                        "prepared": prepared,
                    }
                )
            except Exception as exc:
                finished_at = time.time()
                metrics = _build_metrics(created_at, batch_started_at, finished_at)
                _observe_async_job_metrics(metrics, completed=True)
                JOB_AGE_MS.observe((finished_at - created_at) * 1000.0)
                _print_metrics(job_id, metrics)
                VerifyResultStore.set_done(
                    job_id,
                    {
                        "status": "processing_failed",
                        "liveness_passed": False,
                        "replay_detected": False,
                        "error_code": "invalid_image",
                        "error": str(exc),
                    },
                    metrics,
                )
                redis_client.decr("inflight_jobs")
                _update_inflight_metrics()

        terminal_jobs: list[dict[str, Any]] = []
        ok_jobs = []

        for item in prepared_jobs:
            status = item["prepared"].get("status")

            if status == "ok":
                ok_jobs.append(item)
            else:
                terminal_jobs.append(item)

        for item in terminal_jobs:
            job_id = item["job_id"]
            created_at = item["created_at"]
            prepared = item["prepared"]
            finished_at = time.time()
            metrics = _build_metrics(created_at, batch_started_at, finished_at)
            _observe_async_job_metrics(metrics, completed=True)
            JOB_AGE_MS.observe((finished_at - created_at) * 1000.0)
            _print_metrics(job_id, metrics)

            if prepared["status"] == "quality_reject":
                VerifyResultStore.set_done(
                    job_id,
                    {
                        "status": "quality_reject",
                        "reason": prepared.get("quality_reason"),
                        "quality_details": prepared.get("quality_details", {}),
                        "liveness_passed": None,
                        "replay_detected": False,
                        "error_code": prepared.get("quality_reason") or "quality_reject",
                        "bbox": prepared.get("bbox"),
                        "bbox_source": prepared.get("bbox_source"),
                    },
                    metrics,
                )
            elif prepared["status"] == "spoof":
                VerifyResultStore.set_done(
                    job_id,
                    {
                        "status": "spoof_detected",
                        "liveness_passed": False,
                        "liveness_score": prepared.get("liveness_score"),
                        "replay_detected": False,
                        "error_code": "spoof_detected",
                        "bbox": prepared.get("bbox"),
                        "bbox_source": prepared.get("bbox_source"),
                    },
                    metrics,
                )
            else:
                error_code = prepared.get("error_code")
                if error_code is None and prepared.get("status", "processing_failed") == "processing_failed":
                    error_code = "invalid_image"
                VerifyResultStore.set_done(
                    job_id,
                    {
                        "status": prepared.get("status", "processing_failed"),
                        "liveness_passed": False,
                        "replay_detected": False,
                        "error_code": error_code,
                    },
                    metrics,
                )

            redis_client.decr("inflight_jobs")
            _update_inflight_metrics()

        try:
            if ok_jobs:
                face_inputs = [item["prepared"]["face_input"] for item in ok_jobs]
                t0 = time.time()
                embeddings = pipeline.encoder.encode_batch(face_inputs)
                batch_encode_ms = (time.time() - t0) * 1000.0
                print(f"[BATCH] size={len(ok_jobs)} encode_batch_ms={batch_encode_ms:.2f}")

                if len(embeddings) != len(ok_jobs):
                    raise RuntimeError("Batch encoder returned unexpected batch size")

                for item, embedding in zip(ok_jobs, embeddings):
                    item["prepared"]["embedding"] = embedding

                t0 = time.time()
                async with AsyncSessionLocal() as db:
                    embedding_repo = EmbeddingRepository(db)
                    batch_top_k = await embedding_repo.find_top_k_batch(
                        [item["prepared"]["embedding"] for item in ok_jobs],
                        k=2,
                    )
                batch_search_ms = (time.time() - t0) * 1000.0
                print(f"[BATCH SEARCH] size={len(ok_jobs)} search_ms={batch_search_ms:.2f}")

                for item, top_k in zip(ok_jobs, batch_top_k):
                    item["top_k"] = top_k
        except Exception as exc:
            finished_at = time.time()
            for item in ok_jobs:
                job_id = item["job_id"]
                created_at = item["created_at"]
                metrics = _build_metrics(created_at, batch_started_at, finished_at)
                _observe_async_job_metrics(metrics, completed=True)
                JOB_AGE_MS.observe((finished_at - created_at) * 1000.0)
                _print_metrics(job_id, metrics)
                VerifyResultStore.set_error(job_id, str(exc), metrics)
                redis_client.decr("inflight_jobs")
                _update_inflight_metrics()
            return

        for item in prepared_jobs:
            if item["prepared"].get("status") != "ok":
                continue

            job_id = item["job_id"]
            payload = item["payload"]
            image_bytes = item["image_bytes"]
            created_at = item["created_at"]
            prepared = item["prepared"]
            now = time.time()
            age_ms = (now - created_at) * 1000.0

            if age_ms > MAX_JOB_AGE_MS:
                metrics = _build_metrics(created_at, batch_started_at, now)
                _observe_async_job_metrics(metrics, expired=True)
                JOB_AGE_MS.observe(age_ms)
                _print_metrics(job_id, metrics)
                VERIFY_REJECTED_JOBS.labels(reason="expired").inc()
                VerifyResultStore.set_expired(job_id, metrics)
                redis_client.decr("inflight_jobs")
                _update_inflight_metrics()
                continue

            try:
                async with AsyncSessionLocal() as db:
                    embedding_repo = EmbeddingRepository(db)
                    verification_repo = VerificationRepository(db)
                    search_service = SearchService(embedding_repo)

                    service = VerificationService(
                        embedding_repo=embedding_repo,
                        verification_repo=verification_repo,
                        search_service=search_service,
                        pipeline=pipeline,
                        load_pipeline=False,
                    )

                    result = await service.verify_from_pipeline_result(
                        prepared,
                        image_bytes=image_bytes,
                        user_id=payload.get("user_id"),
                        require_liveness=payload.get("require_liveness", False),
                        check_replay=True,
                        job_id=job_id,
                        t_start=batch_started_at,
                        top_k=item.get("top_k"),
                    )

                finished_at = time.time()
                metrics = _build_metrics(created_at, batch_started_at, finished_at)
                _observe_async_job_metrics(metrics, completed=True)
                JOB_AGE_MS.observe((finished_at - created_at) * 1000.0)
                _print_metrics(job_id, metrics)
                VerifyResultStore.set_done(job_id, result, metrics)
            except Exception as exc:
                finished_at = time.time()
                metrics = _build_metrics(created_at, batch_started_at, finished_at)
                _observe_async_job_metrics(metrics, completed=True)
                JOB_AGE_MS.observe((finished_at - created_at) * 1000.0)
                _print_metrics(job_id, metrics)
                VerifyResultStore.set_error(job_id, str(exc), metrics)
            finally:
                redis_client.decr("inflight_jobs")
                _update_inflight_metrics()


async def run_worker():
    start_http_server(9101)
    await warmup()

    while True:
        batch = await collect_batch()
        if not batch:
            continue

        await process_batch(batch)


if __name__ == "__main__":
    asyncio.run(run_worker())
