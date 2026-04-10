# app/workers/verify_worker.py

import base64
import asyncio
import json
import logging
import os
import time
from time import perf_counter
from typing import Any, cast

import cv2
import numpy as np

import redis

from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.verification_repo import VerificationRepository
from app.core.config import settings
from app.core.logging import setup_logging
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
    ALIGN_CROP_MS,
    DETECT_MS,
    ENCODE_MS,
    PIPELINE_STAGE_DURATION,
    LIVENESS_MS,
    PREPROCESS_MS,
    RESULT_WRITE_MS,
    QUALITY_REJECT_COUNTER,
    QUALITY_GATE_PRE_MS,
    QUALITY_GATE_FACE_MS,
    REDIS_COMMAND_LATENCY_MS,
    QUEUE_POP_LATENCY_MS,
    VECTOR_SEARCH_MS,
    VERIFY_INFLIGHT_CURRENT,
    VERIFY_REJECTED_JOBS,
    VERIFY_WORKER_UTILIZATION,
)
from prometheus_client import start_http_server
from app.services.search_service import SearchService
from app.services.verify_job_queue import VerifyJobQueue
from app.services.verify_result_store import VerifyResultStore
from app.services.verification_service import VerificationService

logger = logging.getLogger(__name__)
METRICS_ENABLED = True

redis_client = redis.from_url(settings.redis_url, decode_responses=True)
QUEUE_NAME = "face_verify_queue"
MAX_QUEUE_WAIT = 5.0
MAX_JOB_AGE_MS = 3000
MAX_IMAGE_SIDE = 480
BATCH_COLLECT_TIMEOUT = float(os.getenv("BATCH_COLLECT_TIMEOUT", "0.005"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
_PIPELINE: Any | None = None

semaphore = asyncio.Semaphore(max(1, int(settings.WORKER_SEMAPHORE)))

DECR_SCRIPT = """
local delta = tonumber(ARGV[1])
local val = redis.call("DECRBY", KEYS[1], delta)
if val < 0 then
    redis.call("SET", KEYS[1], 0)
    return 0
end
return val
"""


def _decr_inflight(count: int = 1) -> int:
    start = perf_counter()
    result = cast(Any, redis_client.eval(DECR_SCRIPT, 1, "inflight_jobs", str(count)))
    REDIS_COMMAND_LATENCY_MS.labels(command="eval_decr_inflight_jobs").observe(
        (perf_counter() - start) * 1000.0
    )
    return int(result)


def _update_inflight_metrics() -> None:
    start = perf_counter()
    inflight_raw = cast(str | None, redis_client.get("inflight_jobs"))
    REDIS_COMMAND_LATENCY_MS.labels(command="get_inflight_jobs").observe(
        (perf_counter() - start) * 1000.0
    )
    inflight = int(inflight_raw) if inflight_raw else 0
    VERIFY_INFLIGHT_CURRENT.set(inflight)
    VERIFY_WORKER_UTILIZATION.set(
        min(1.0, inflight / float(VerifyJobQueue.INFLIGHT_LIMIT))
    )


def _update_queue_length_metric() -> None:
    start = perf_counter()
    queue_length = float(cast(int, redis_client.llen(QUEUE_NAME)))
    REDIS_COMMAND_LATENCY_MS.labels(command="llen_queue").observe(
        (perf_counter() - start) * 1000.0
    )
    QUEUE_LENGTH.set(queue_length)


def _observe_worker_stage(stage: str, duration_ms: float) -> None:
    if not METRICS_ENABLED or duration_ms <= 0:
        return

    try:
        # PIPELINE_STAGE_DURATION хранится в секундах
        PIPELINE_STAGE_DURATION.labels(stage=stage).observe(float(duration_ms) / 1000.0)
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


def _extract_stage_timings(prepared_timings: dict[str, Any] | None) -> dict[str, float]:
    timings = prepared_timings or {}

    detect_ms = float(
        timings.get(
            "detect_ms",
            timings.get("fast_detect_ms", 0.0) + timings.get("fallback_detect_ms", 0.0),
        )
    )

    return {
        "b64_decode_ms": float(timings.get("b64_decode_ms", 0.0)),
        "image_decode_ms": float(timings.get("image_decode_ms", 0.0)),
        "downscale_ms": float(timings.get("downscale_ms", 0.0)),
        "jpeg_reencode_ms": float(timings.get("jpeg_reencode_ms", 0.0)),
        "batch_collect_wait_ms": float(timings.get("batch_collect_wait_ms", 0.0)),
        "batch_prepare_wall_ms": float(timings.get("batch_prepare_wall_ms", 0.0)),
        "batch_encode_wall_ms": float(timings.get("batch_encode_wall_ms", 0.0)),
        "batch_search_wall_ms": float(timings.get("batch_search_wall_ms", 0.0)),
        "batch_verify_loop_wait_ms": float(timings.get("batch_verify_loop_wait_ms", 0.0)),
        "preprocess_ms": float(timings.get("preprocess_ms", 0.0)),
        "quality_gate_pre_ms": float(timings.get("quality_gate_pre_ms", 0.0)),
        "detect_ms": detect_ms,
        "detect_blob_ms": float(timings.get("detect_blob_ms", 0.0)),
        "detect_forward_ms": float(timings.get("detect_forward_ms", 0.0)),
        "detect_decode_ms": float(timings.get("detect_decode_ms", 0.0)),
        "align_crop_ms": float(timings.get("align_crop_ms", 0.0)),
        "quality_gate_face_ms": float(timings.get("quality_gate_face_ms", 0.0)),
        "liveness_ms": float(timings.get("liveness_ms", 0.0)),
        "encode_ms": float(timings.get("encode_ms", 0.0)),
        "encode_preprocess_ms": float(timings.get("encode_preprocess_ms", 0.0)),
        "encode_ort_run_ms": float(timings.get("encode_ort_run_ms", 0.0)),
        "encode_postprocess_ms": float(timings.get("encode_postprocess_ms", 0.0)),
        "vector_search_ms": float(timings.get("vector_search_ms", 0.0)),
        "anti_replay_ms": float(timings.get("anti_replay_ms", 0.0)),
        "is_genuine_ms": float(timings.get("is_genuine_ms", 0.0)),
        "decision_ms": float(timings.get("decision_ms", 0.0)),
        "verification_log_write_ms": float(timings.get("verification_log_write_ms", 0.0)),
        "verify_from_pipeline_result_ms": float(timings.get("verify_from_pipeline_result_ms", 0.0)),
    }


def _observe_prepared_timings(prepared_timings: dict[str, Any] | None) -> None:
    if not METRICS_ENABLED:
        return

    stage_timings = _extract_stage_timings(prepared_timings)

    try:
        for stage_name in (
            "b64_decode_ms",
            "image_decode_ms",
            "downscale_ms",
            "jpeg_reencode_ms",
            "batch_collect_wait_ms",
            "batch_prepare_wall_ms",
            "batch_encode_wall_ms",
            "batch_search_wall_ms",
            "batch_verify_loop_wait_ms",
            "detect_blob_ms",
            "detect_forward_ms",
            "detect_decode_ms",
            "encode_preprocess_ms",
            "encode_ort_run_ms",
            "encode_postprocess_ms",
            "anti_replay_ms",
            "is_genuine_ms",
            "decision_ms",
            "verification_log_write_ms",
            "verify_from_pipeline_result_ms",
        ):
            if stage_timings[stage_name] > 0:
                _observe_worker_stage(stage_name.removesuffix("_ms"), stage_timings[stage_name])

        if stage_timings["preprocess_ms"] > 0:
            PREPROCESS_MS.observe(stage_timings["preprocess_ms"])
            _observe_worker_stage("preprocess", stage_timings["preprocess_ms"])

        if stage_timings["quality_gate_pre_ms"] > 0:
            QUALITY_GATE_PRE_MS.observe(stage_timings["quality_gate_pre_ms"])
            _observe_worker_stage("quality_gate_pre", stage_timings["quality_gate_pre_ms"])

        if stage_timings["detect_ms"] > 0:
            DETECT_MS.observe(stage_timings["detect_ms"])
            _observe_worker_stage("detect", stage_timings["detect_ms"])

        if stage_timings["align_crop_ms"] > 0:
            ALIGN_CROP_MS.observe(stage_timings["align_crop_ms"])
            _observe_worker_stage("align_crop", stage_timings["align_crop_ms"])

        if stage_timings["quality_gate_face_ms"] > 0:
            QUALITY_GATE_FACE_MS.observe(stage_timings["quality_gate_face_ms"])
            _observe_worker_stage("quality_gate_face", stage_timings["quality_gate_face_ms"])

        if stage_timings["liveness_ms"] > 0:
            LIVENESS_MS.observe(stage_timings["liveness_ms"])
            _observe_worker_stage("liveness", stage_timings["liveness_ms"])

        if stage_timings["encode_ms"] > 0:
            ENCODE_MS.observe(stage_timings["encode_ms"])
            _observe_worker_stage("encode", stage_timings["encode_ms"])

        if stage_timings["vector_search_ms"] > 0:
            VECTOR_SEARCH_MS.observe(stage_timings["vector_search_ms"])
            _observe_worker_stage("vector_search", stage_timings["vector_search_ms"])
    except Exception:
        pass


def _timed_result_write(write_fn, *args, **kwargs) -> float:
    start = perf_counter()
    write_fn(*args, **kwargs)
    duration_ms = (perf_counter() - start) * 1000.0
    _observe_result_write(duration_ms)
    return duration_ms


def _log_stage_times(
    *,
    job_id: str,
    created_at: float,
    job_started_at: float,
    finished_at: float,
    prepared_timings: dict[str, Any] | None = None,
    result_write_ms: float | None = None,
    vector_search_ms: float | None = None,
    dequeued_at: float | None = None,
    outcome: str | None = None,
    quality_reason: str | None = None,
    quality_details: dict[str, Any] | None = None,
) -> None:
    stage_timings = _extract_stage_timings(prepared_timings)
    if vector_search_ms is not None:
        stage_timings["vector_search_ms"] = float(vector_search_ms)

    quality_details = quality_details or {}
    image_width = quality_details.get("image_width")
    image_height = quality_details.get("image_height")
    blur_score = quality_details.get("blur_score")
    brightness = quality_details.get("brightness")
    contrast = quality_details.get("contrast")
    face_width = quality_details.get("face_width")
    face_height = quality_details.get("face_height")
    min_face_side = quality_details.get("min_face_side")
    quality_stage = quality_details.get("quality_stage")
    quality_mode = quality_details.get("quality_gate_mode")
    quality_warning = quality_details.get("quality_warning")

    result_write_value = float(result_write_ms or 0.0)
    queue_delay_ms = max(
        0.0,
        (((dequeued_at if dequeued_at is not None else job_started_at) - created_at) * 1000.0),
    )
    dequeue_to_start_ms = (
        max(0.0, (job_started_at - dequeued_at) * 1000.0) if dequeued_at is not None else 0.0
    )
    processing_ms = max(0.0, (finished_at - job_started_at) * 1000.0)
    e2e_ms = max(0.0, (finished_at - created_at) * 1000.0)

    accounted_ms = (
        stage_timings["b64_decode_ms"]
        + stage_timings["image_decode_ms"]
        + stage_timings["downscale_ms"]
        + stage_timings["jpeg_reencode_ms"]
        + stage_timings["batch_collect_wait_ms"]
        + stage_timings["batch_verify_loop_wait_ms"]
        + stage_timings["preprocess_ms"]
        + stage_timings["quality_gate_pre_ms"]
        + stage_timings["detect_ms"]
        + stage_timings["align_crop_ms"]
        + stage_timings["quality_gate_face_ms"]
        + stage_timings["liveness_ms"]
        + stage_timings["encode_ms"]
        + stage_timings["vector_search_ms"]
        + stage_timings["anti_replay_ms"]
        + stage_timings["is_genuine_ms"]
        + stage_timings["decision_ms"]
        + stage_timings["verification_log_write_ms"]
        + result_write_value
    )
    unattributed_ms = max(0.0, processing_ms - accounted_ms)

    logger.warning(
        "stage_times job_id=%s outcome=%s quality_reason=%s quality_stage=%s quality_mode=%s quality_warning=%s "
        "image_width=%s image_height=%s blur_score=%s brightness=%s contrast=%s "
        "face_width=%s face_height=%s min_face_side=%s "
        "queue_delay_ms=%.3f dequeue_to_start_ms=%.3f preprocess_ms=%.3f quality_gate_pre_ms=%.3f "
        "batch_collect_wait_ms=%.3f batch_prepare_wall_ms=%.3f batch_encode_wall_ms=%.3f "
        "batch_search_wall_ms=%.3f batch_verify_loop_wait_ms=%.3f "
        "detect_ms=%.3f detect_blob_ms=%.3f detect_forward_ms=%.3f detect_decode_ms=%.3f "
        "align_crop_ms=%.3f quality_gate_face_ms=%.3f liveness_ms=%.3f "
        "encode_ms=%.3f encode_preprocess_ms=%.3f encode_ort_run_ms=%.3f encode_postprocess_ms=%.3f "
        "vector_search_ms=%.3f anti_replay_ms=%.3f is_genuine_ms=%.3f "
        "decision_ms=%.3f verification_log_write_ms=%.3f verify_from_pipeline_result_ms=%.6f "
        "result_write_ms=%.3f processing_ms=%.3f unattributed_ms=%.3f e2e_ms=%.3f faiss_enabled=%s",
        job_id,
        outcome or "unknown",
        quality_reason,
        quality_stage,
        quality_mode,
        quality_warning,
        image_width,
        image_height,
        blur_score,
        brightness,
        contrast,
        face_width,
        face_height,
        min_face_side,
        queue_delay_ms,
        dequeue_to_start_ms,
        stage_timings["preprocess_ms"],
        stage_timings["quality_gate_pre_ms"],
        stage_timings["batch_collect_wait_ms"],
        stage_timings["batch_prepare_wall_ms"],
        stage_timings["batch_encode_wall_ms"],
        stage_timings["batch_search_wall_ms"],
        stage_timings["batch_verify_loop_wait_ms"],
        stage_timings["detect_ms"],
        stage_timings["detect_blob_ms"],
        stage_timings["detect_forward_ms"],
        stage_timings["detect_decode_ms"],
        stage_timings["align_crop_ms"],
        stage_timings["quality_gate_face_ms"],
        stage_timings["liveness_ms"],
        stage_timings["encode_ms"],
        stage_timings["encode_preprocess_ms"],
        stage_timings["encode_ort_run_ms"],
        stage_timings["encode_postprocess_ms"],
        stage_timings["vector_search_ms"],
        stage_timings["anti_replay_ms"],
        stage_timings["is_genuine_ms"],
        stage_timings["decision_ms"],
        stage_timings["verification_log_write_ms"],
        stage_timings["verify_from_pipeline_result_ms"],
        result_write_value,
        processing_ms,
        unattributed_ms,
        e2e_ms,
        bool(settings.FAISS_ENABLED),
    )
    print(
        "stage_times "
        f"job_id={job_id} "
        f"outcome={outcome or 'unknown'} "
        f"quality_reason={quality_reason} "
        f"quality_stage={quality_stage} "
        f"quality_mode={quality_mode} "
        f"quality_warning={quality_warning} "
        f"image_width={image_width} "
        f"image_height={image_height} "
        f"blur_score={blur_score} "
        f"brightness={brightness} "
        f"contrast={contrast} "
        f"face_width={face_width} "
        f"face_height={face_height} "
        f"min_face_side={min_face_side} "
        f"queue_delay_ms={queue_delay_ms:.3f} "
        f"dequeue_to_start_ms={dequeue_to_start_ms:.3f} "
        f"preprocess_ms={stage_timings['preprocess_ms']:.3f} "
        f"quality_gate_pre_ms={stage_timings['quality_gate_pre_ms']:.3f} "
        f"batch_collect_wait_ms={stage_timings['batch_collect_wait_ms']:.3f} "
        f"batch_prepare_wall_ms={stage_timings['batch_prepare_wall_ms']:.3f} "
        f"batch_encode_wall_ms={stage_timings['batch_encode_wall_ms']:.3f} "
        f"batch_search_wall_ms={stage_timings['batch_search_wall_ms']:.3f} "
        f"batch_verify_loop_wait_ms={stage_timings['batch_verify_loop_wait_ms']:.3f} "
        f"detect_ms={stage_timings['detect_ms']:.3f} "
        f"detect_blob_ms={stage_timings['detect_blob_ms']:.3f} "
        f"detect_forward_ms={stage_timings['detect_forward_ms']:.3f} "
        f"detect_decode_ms={stage_timings['detect_decode_ms']:.3f} "
        f"align_crop_ms={stage_timings['align_crop_ms']:.3f} "
        f"quality_gate_face_ms={stage_timings['quality_gate_face_ms']:.3f} "
        f"liveness_ms={stage_timings['liveness_ms']:.3f} "
        f"encode_ms={stage_timings['encode_ms']:.3f} "
        f"encode_preprocess_ms={stage_timings['encode_preprocess_ms']:.3f} "
        f"encode_ort_run_ms={stage_timings['encode_ort_run_ms']:.3f} "
        f"encode_postprocess_ms={stage_timings['encode_postprocess_ms']:.3f} "
        f"vector_search_ms={stage_timings['vector_search_ms']:.3f} "
        f"anti_replay_ms={stage_timings['anti_replay_ms']:.3f} "
        f"is_genuine_ms={stage_timings['is_genuine_ms']:.3f} "
        f"decision_ms={stage_timings['decision_ms']:.3f} "
        f"verification_log_write_ms={stage_timings['verification_log_write_ms']:.3f} "
        f"verify_from_pipeline_result_ms={stage_timings['verify_from_pipeline_result_ms']:.6f} "
        f"result_write_ms={result_write_value:.3f} "
        f"processing_ms={processing_ms:.3f} "
        f"unattributed_ms={unattributed_ms:.3f} "
        f"e2e_ms={e2e_ms:.3f} "
        f"faiss_enabled={bool(settings.FAISS_ENABLED)}",
        flush=True,
    )


def _build_metrics(
    created_at: float,
    started_at: float,
    finished_at: float,
    *,
    dequeued_at: float | None = None,
) -> dict[str, float]:
    effective_dequeued_at = dequeued_at if dequeued_at is not None else started_at

    return {
        "created_at": created_at,
        "dequeued_at": effective_dequeued_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "queue_delay": max(0.0, effective_dequeued_at - created_at),
        "dequeue_to_start": max(0.0, started_at - effective_dequeued_at),
        "processing_time": max(0.0, finished_at - started_at),
        "total_latency": max(0.0, finished_at - created_at),
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
    start = perf_counter()
    redis_client.setex(
        "metrics:queue_delay_ms",
        5,
        str(queue_delay_ms),
    )
    REDIS_COMMAND_LATENCY_MS.labels(command="setex_queue_delay_ms").observe(
        (perf_counter() - start) * 1000.0
    )

    if completed:
        ASYNC_JOB_COMPLETED_TOTAL.inc()
    if expired:
        ASYNC_JOB_EXPIRED_TOTAL.inc()


def _print_metrics(job_id: str, metrics: dict[str, float]) -> None:
    print(
        f"[METRICS] job={job_id} "
        f"queue={metrics['queue_delay']:.2f}s "
        f"dq_wait={metrics.get('dequeue_to_start', 0.0):.2f}s "
        f"proc={metrics['processing_time']:.2f}s "
        f"total={metrics['total_latency']:.2f}s"
    )


def _finish_expired_job(
    job_id: str,
    created_at: float,
    started_at: float,
    *,
    dequeued_at: float | None = None,
) -> None:
    write_metrics = _build_metrics(
        created_at,
        started_at,
        time.time(),
        dequeued_at=dequeued_at if dequeued_at is not None else started_at,
    )
    _timed_result_write(VerifyResultStore.set_expired, job_id, write_metrics)
    finished_at = time.time()
    metrics = _build_metrics(
        created_at,
        started_at,
        finished_at,
        dequeued_at=dequeued_at if dequeued_at is not None else started_at,
    )
    age_ms = (finished_at - created_at) * 1000.0
    _observe_async_job_metrics(metrics, expired=True)
    JOB_AGE_MS.observe(age_ms)
    _print_metrics(job_id, metrics)
    VERIFY_REJECTED_JOBS.labels(reason="expired").inc()
    _decr_inflight()
    _update_inflight_metrics()


def _reject_stale_job(job_id: str, created_at: float, observed_at: float) -> None:
    write_metrics = _build_metrics(created_at, observed_at, time.time(), dequeued_at=observed_at)
    _timed_result_write(VerifyResultStore.set_expired, job_id, write_metrics)
    finished_at = time.time()
    metrics = _build_metrics(created_at, observed_at, finished_at, dequeued_at=observed_at)
    age_ms = (finished_at - created_at) * 1000.0
    _observe_async_job_metrics(metrics, expired=True)
    JOB_AGE_MS.observe(age_ms)
    _print_metrics(job_id, metrics)
    VERIFY_REJECTED_JOBS.labels(reason="expired").inc()
    _decr_inflight()
    _update_inflight_metrics()


def _decode_and_downscale_image(
    image_bytes: bytes,
    max_side: int = MAX_IMAGE_SIDE,
) -> tuple[np.ndarray | None, dict[str, float]]:
    timings: dict[str, float] = {}

    t0 = perf_counter()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    timings["image_decode_ms"] = (perf_counter() - t0) * 1000.0

    if decoded is None:
        timings["downscale_ms"] = 0.0
        timings["jpeg_reencode_ms"] = 0.0
        return None, timings

    height, width = decoded.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        timings["downscale_ms"] = 0.0
        timings["jpeg_reencode_ms"] = 0.0
        return decoded, timings

    scale = max_side / float(longest_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    t0 = perf_counter()
    resized = cv2.resize(decoded, (new_width, new_height), interpolation=cv2.INTER_AREA)
    timings["downscale_ms"] = (perf_counter() - t0) * 1000.0
    timings["jpeg_reencode_ms"] = 0.0

    return resized, timings


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
    start = perf_counter()
    redis_client.llen(QUEUE_NAME)
    REDIS_COMMAND_LATENCY_MS.labels(command="llen_queue_pre_batch").observe(
        (perf_counter() - start) * 1000.0
    )
    _update_queue_length_metric()

    jobs: list[dict[str, Any]] = []
    batch_size = BATCH_SIZE

    first_job: dict[str, Any] | None = None
    while first_job is None:
        start = perf_counter()
        raw = cast(
            tuple[str, str] | None,
            redis_client.brpop([QUEUE_NAME], timeout=0),  # type: ignore[arg-type]
        )
        QUEUE_POP_LATENCY_MS.observe((perf_counter() - start) * 1000.0)
        if raw is None:
            return []

        _, data = raw
        job = json.loads(data)
        job["dequeued_at"] = time.time()
        created_at = job.get("created_at", time.time())
        now = time.time()
        age_ms = (now - created_at) * 1000.0

        if age_ms > MAX_JOB_AGE_MS or (now - created_at) > MAX_QUEUE_WAIT:
            _reject_stale_job(job["job_id"], created_at, now)
            continue

        first_job = job
        jobs.append(job)

    while len(jobs) < batch_size:
        start = perf_counter()
        raw = cast(str | None, redis_client.lpop(QUEUE_NAME))
        QUEUE_POP_LATENCY_MS.observe((perf_counter() - start) * 1000.0)
        if raw is None:
            break

        job = json.loads(raw)
        job["dequeued_at"] = time.time()
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
    pipeline = _PIPELINE
    if pipeline is None:
        raise RuntimeError("Pipeline is required for batch processing")

    batch_started_at = time.time()
    prepared_jobs: list[dict[str, Any]] = []
    batch_candidates: list[dict[str, Any]] = []
    inflight_decrements = 0

    for job_data in job_datas:
        job_started_at = time.time()
        job_id = job_data["job_id"]
        payload = job_data["payload"]
        created_at = job_data.get("created_at", batch_started_at)
        dequeued_at = float(job_data.get("dequeued_at", job_started_at))
        now = time.time()
        age_ms = (now - created_at) * 1000.0

        if age_ms > MAX_JOB_AGE_MS:
            metrics = _build_metrics(created_at, job_started_at, now, dequeued_at=dequeued_at)
            _observe_async_job_metrics(metrics, expired=True)
            JOB_AGE_MS.observe(age_ms)
            _print_metrics(job_id, metrics)
            VERIFY_REJECTED_JOBS.labels(reason="expired").inc()
            _timed_result_write(VerifyResultStore.set_expired, job_id, metrics)
            inflight_decrements += 1
            continue

        if now - created_at > MAX_QUEUE_WAIT:
            metrics = _build_metrics(created_at, job_started_at, now, dequeued_at=dequeued_at)
            _observe_async_job_metrics(metrics, expired=True)
            JOB_AGE_MS.observe(age_ms)
            _print_metrics(job_id, metrics)
            _timed_result_write(VerifyResultStore.set_expired, job_id, metrics)
            inflight_decrements += 1
            continue

        if (batch_started_at - created_at) > MAX_QUEUE_WAIT:
            _finish_expired_job(job_id, created_at, job_started_at, dequeued_at=dequeued_at)
            continue

        try:
            t0 = perf_counter()
            original_image_bytes = base64.b64decode(payload["image_b64"])
            b64_decode_ms = (perf_counter() - t0) * 1000.0

            image, worker_pre_timings = _decode_and_downscale_image(original_image_bytes)
            worker_pre_timings["b64_decode_ms"] = b64_decode_ms
            worker_pre_timings["batch_collect_wait_ms"] = max(
                0.0,
                (batch_started_at - dequeued_at) * 1000.0,
            )

            if image is None:
                raise ValueError("invalid_image")

            batch_candidates.append(
                {
                    "job_id": job_id,
                    "payload": payload,
                    "original_image_bytes": original_image_bytes,
                    "image_bytes": original_image_bytes,
                    "image": image,
                    "created_at": created_at,
                    "dequeued_at": dequeued_at,
                    "job_started_at": job_started_at,
                    "worker_pre_timings": worker_pre_timings,
                }
            )
        except Exception as exc:
            write_metrics = _build_metrics(created_at, job_started_at, time.time(), dequeued_at=dequeued_at)
            result_write_ms = _timed_result_write(
                VerifyResultStore.set_done,
                job_id,
                {
                    "status": "processing_failed",
                    "liveness_passed": False,
                    "replay_detected": False,
                    "error_code": "invalid_image",
                    "error": str(exc),
                },
                write_metrics,
            )
            finished_at = time.time()
            metrics = _build_metrics(created_at, job_started_at, finished_at, dequeued_at=dequeued_at)
            _observe_async_job_metrics(metrics, completed=True)
            JOB_AGE_MS.observe((finished_at - created_at) * 1000.0)
            _print_metrics(job_id, metrics)
            metrics["result_write_ms"] = result_write_ms
            inflight_decrements += 1

    if batch_candidates:
        t0 = perf_counter()
        try:
            decoded_images = [item.get("image") for item in batch_candidates]
            if all(image is not None for image in decoded_images):
                prepared_results = pipeline.prepare_face_inputs_from_images(
                    [cast(np.ndarray, image) for image in decoded_images]
                )
            else:
                image_bytes_list = [item["image_bytes"] for item in batch_candidates]
                prepared_results = pipeline.prepare_face_inputs(image_bytes_list)
            prep_ms = (perf_counter() - t0) * 1000.0
            logger.info(
                "[PREP_BATCH] size=%s prep_ms=%.2f",
                len(batch_candidates),
                prep_ms,
            )

            if len(prepared_results) != len(batch_candidates):
                raise RuntimeError("Pipeline returned unexpected batch size")

            detector = getattr(pipeline, "fast_detector", None)
            detector_batch_timings = getattr(detector, "last_batch_timings", {}) or {}

            for item, prepared in zip(batch_candidates, prepared_results):
                prepared_timings = prepared.setdefault("timings", {})
                prepared_timings.update(item.get("worker_pre_timings", {}))
                prepared_timings["batch_prepare_wall_ms"] = prep_ms

                if float(prepared_timings.get("detect_ms", 0.0)) > 0.0:
                    prepared_timings["detect_blob_ms"] = float(
                        detector_batch_timings.get("detect_blob_ms_per_image", 0.0)
                    )
                    prepared_timings["detect_forward_ms"] = float(
                        detector_batch_timings.get("detect_forward_ms_per_image", 0.0)
                    )
                    prepared_timings["detect_decode_ms"] = float(
                        detector_batch_timings.get("detect_decode_ms_per_image", 0.0)
                    )
                    prepared_timings["detect_batch_fallback"] = bool(
                        detector_batch_timings.get("detect_batch_fallback", False)
                    )

                item["prepared"] = prepared
                prepared_jobs.append(item)
        except Exception as exc:
            finished_at = time.time()
            for item in batch_candidates:
                job_id = item["job_id"]
                created_at = item["created_at"]
                job_started_at = item["job_started_at"]
                dequeued_at = item.get("dequeued_at", job_started_at)
                write_metrics = _build_metrics(created_at, job_started_at, time.time(), dequeued_at=dequeued_at)
                _timed_result_write(VerifyResultStore.set_error, job_id, str(exc), write_metrics)
                finished_at = time.time()
                metrics = _build_metrics(created_at, job_started_at, finished_at, dequeued_at=dequeued_at)
                _observe_async_job_metrics(metrics, completed=True)
                JOB_AGE_MS.observe((finished_at - created_at) * 1000.0)
                _print_metrics(job_id, metrics)
                inflight_decrements += 1
            if inflight_decrements:
                _decr_inflight(inflight_decrements)
                _update_inflight_metrics()
            return

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
        job_started_at = item["job_started_at"]
        prepared = item["prepared"]
        dequeued_at = item.get("dequeued_at", job_started_at)
        prepared_timings = prepared.get("timings", {})
        _observe_prepared_timings(prepared_timings)

        if prepared["status"] == "quality_reject":
            QUALITY_REJECT_COUNTER.labels(
                reason=prepared.get("quality_reason", "unknown")
            ).inc()
            write_metrics = _build_metrics(created_at, job_started_at, time.time(), dequeued_at=dequeued_at)
            result_write_ms = _timed_result_write(
                VerifyResultStore.set_done,
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
                    "bbox_source_detail": prepared.get("bbox_source_detail"),
                },
                write_metrics,
            )
            finished_at = time.time()
            metrics = _build_metrics(created_at, job_started_at, finished_at, dequeued_at=dequeued_at)
            _observe_async_job_metrics(metrics, completed=True)
            JOB_AGE_MS.observe((finished_at - created_at) * 1000.0)
            _print_metrics(job_id, metrics)
            metrics["result_write_ms"] = result_write_ms
            _log_stage_times(
                job_id=job_id,
                created_at=created_at,
                job_started_at=job_started_at,
                finished_at=finished_at,
                prepared_timings=prepared.get("timings", {}),
                result_write_ms=result_write_ms,
                dequeued_at=item.get("dequeued_at"),
                outcome="quality_reject",
                quality_reason=prepared.get("quality_reason"),
                quality_details=prepared.get("quality_details", {}),
            )
        elif prepared["status"] == "spoof":
            write_metrics = _build_metrics(created_at, job_started_at, time.time(), dequeued_at=dequeued_at)
            result_write_ms = _timed_result_write(
                VerifyResultStore.set_done,
                job_id,
                {
                    "status": "spoof_detected",
                    "liveness_passed": False,
                    "liveness_score": prepared.get("liveness_score"),
                    "replay_detected": False,
                    "error_code": "spoof_detected",
                    "bbox": prepared.get("bbox"),
                    "bbox_source": prepared.get("bbox_source"),
                    "bbox_source_detail": prepared.get("bbox_source_detail"),
                },
                write_metrics,
            )
            finished_at = time.time()
            metrics = _build_metrics(created_at, job_started_at, finished_at, dequeued_at=dequeued_at)
            _observe_async_job_metrics(metrics, completed=True)
            JOB_AGE_MS.observe((finished_at - created_at) * 1000.0)
            _print_metrics(job_id, metrics)
            metrics["result_write_ms"] = result_write_ms
            _log_stage_times(
                job_id=job_id,
                created_at=created_at,
                job_started_at=job_started_at,
                finished_at=finished_at,
                prepared_timings=prepared.get("timings", {}),
                result_write_ms=result_write_ms,
                dequeued_at=item.get("dequeued_at"),
                outcome="spoof",
                quality_reason=prepared.get("quality_reason"),
                quality_details=prepared.get("quality_details", {}),
            )
        else:
            error_code = prepared.get("error_code")
            if error_code is None and prepared.get("status", "processing_failed") == "processing_failed":
                error_code = "invalid_image"
            write_metrics = _build_metrics(created_at, job_started_at, time.time(), dequeued_at=dequeued_at)
            result_write_ms = _timed_result_write(
                VerifyResultStore.set_done,
                job_id,
                {
                    "status": prepared.get("status", "processing_failed"),
                    "liveness_passed": False,
                    "replay_detected": False,
                    "error_code": error_code,
                    "bbox_source": prepared.get("bbox_source"),
                    "bbox_source_detail": prepared.get("bbox_source_detail"),
                },
                write_metrics,
            )
            finished_at = time.time()
            metrics = _build_metrics(created_at, job_started_at, finished_at, dequeued_at=dequeued_at)
            _observe_async_job_metrics(metrics, completed=True)
            JOB_AGE_MS.observe((finished_at - created_at) * 1000.0)
            _print_metrics(job_id, metrics)
            metrics["result_write_ms"] = result_write_ms
            _log_stage_times(
                job_id=job_id,
                created_at=created_at,
                job_started_at=job_started_at,
                finished_at=finished_at,
                prepared_timings=prepared.get("timings", {}),
                result_write_ms=result_write_ms,
                dequeued_at=item.get("dequeued_at"),
                outcome="processing_failed",
                quality_reason=prepared.get("quality_reason"),
                quality_details=prepared.get("quality_details", {}),
            )

        inflight_decrements += 1

    try:
        if ok_jobs:
            face_inputs = [item["prepared"]["face_input"] for item in ok_jobs]
            t0 = perf_counter()
            async with semaphore:
                embeddings = pipeline.encoder.encode_batch(face_inputs)
            batch_encode_ms = (perf_counter() - t0) * 1000.0
            logger.info(
                "[BATCH] size=%s encode_batch_ms=%.2f",
                len(ok_jobs),
                batch_encode_ms,
            )

            if len(embeddings) != len(ok_jobs):
                raise RuntimeError("Batch encoder returned unexpected batch size")

            for item, embedding in zip(ok_jobs, embeddings):
                item["prepared"]["embedding"] = embedding

            estimated_encode_ms = batch_encode_ms / max(1, len(ok_jobs))

            encoder_wrapper = getattr(pipeline, "encoder", None)
            encoder_impl = getattr(encoder_wrapper, "encoder", encoder_wrapper)
            encoder_batch_timings = getattr(encoder_impl, "last_batch_timings", {}) or {}

            for item in ok_jobs:
                job_id = item["job_id"]
                prepared = item["prepared"]
                prepared_timings = prepared.get("timings", {})
                prepared_timings["encode_ms"] = estimated_encode_ms
                prepared_timings["batch_encode_wall_ms"] = batch_encode_ms
                prepared_timings["encode_preprocess_ms"] = float(
                    encoder_batch_timings.get("encode_preprocess_ms_per_image", 0.0)
                )
                prepared_timings["encode_ort_run_ms"] = float(
                    encoder_batch_timings.get("encode_ort_run_ms_per_image", 0.0)
                )
                prepared_timings["encode_postprocess_ms"] = float(
                    encoder_batch_timings.get("encode_postprocess_ms_per_image", 0.0)
                )
                if prepared_timings:
                    print(
                        "[PIPELINE] "
                        f"job={job_id} "
                        f"pre={prepared_timings.get('preprocess_ms', 0):.2f} "
                        f"qpre={prepared_timings.get('quality_gate_pre_ms', 0):.2f} "
                        f"detect={prepared_timings.get('detect_ms', 0):.2f} "
                        f"detect_fwd={prepared_timings.get('detect_forward_ms', 0):.2f} "
                        f"crop={prepared_timings.get('align_crop_ms', 0):.2f} "
                        f"qface={prepared_timings.get('quality_gate_face_ms', 0):.2f} "
                        f"live={prepared_timings.get('liveness_ms', 0):.2f} "
                        f"enc_ort={prepared_timings.get('encode_ort_run_ms', 0):.2f}"
                    , flush=True)

                print(
                    "[ENCODE] "
                    f"job={job_id} "
                    f"encode_ms={prepared_timings.get('encode_ms', estimated_encode_ms):.2f}"
                , flush=True)

            t0 = perf_counter()
            async with AsyncSessionLocal() as db:
                embedding_repo = EmbeddingRepository(db)
                batch_top_k = await embedding_repo.find_top_k_batch(
                    [item["prepared"]["embedding"] for item in ok_jobs],
                    k=2,
                )
            batch_search_ms = (perf_counter() - t0) * 1000.0
            print(f"[BATCH SEARCH] size={len(ok_jobs)} search_ms={batch_search_ms:.2f}")
            estimated_vector_search_ms = batch_search_ms / max(1, len(ok_jobs))

            for item, top_k in zip(ok_jobs, batch_top_k):
                item["top_k"] = top_k
                item["prepared"]["timings"]["vector_search_ms"] = estimated_vector_search_ms
                item["prepared"]["timings"]["batch_search_wall_ms"] = batch_search_ms

            logger.info(
                "[BATCH] size=%s ready=%s",
                len(prepared_jobs),
                len(ok_jobs),
            )
    except Exception as exc:
        for item in ok_jobs:
            job_id = item["job_id"]
            created_at = item["created_at"]
            job_started_at = item["job_started_at"]
            dequeued_at = item.get("dequeued_at", job_started_at)
            prepared = item["prepared"]
            prepared_timings = prepared.get("timings", {})
            _observe_prepared_timings(prepared_timings)
            write_metrics = _build_metrics(created_at, job_started_at, time.time(), dequeued_at=dequeued_at)
            result_write_ms = _timed_result_write(VerifyResultStore.set_error, job_id, str(exc), write_metrics)
            finished_at = time.time()
            metrics = _build_metrics(created_at, job_started_at, finished_at, dequeued_at=dequeued_at)
            _observe_async_job_metrics(metrics, completed=True)
            JOB_AGE_MS.observe((finished_at - created_at) * 1000.0)
            _print_metrics(job_id, metrics)
            metrics["result_write_ms"] = result_write_ms
            _log_stage_times(
                job_id=job_id,
                created_at=created_at,
                job_started_at=job_started_at,
                finished_at=finished_at,
                prepared_timings=prepared_timings,
                result_write_ms=result_write_ms,
                vector_search_ms=prepared_timings.get("vector_search_ms"),
                dequeued_at=dequeued_at,
                outcome="error",
                quality_reason=prepared.get("quality_reason"),
                quality_details=prepared.get("quality_details", {}),
            )
            inflight_decrements += 1
        if inflight_decrements:
            _decr_inflight(inflight_decrements)
            _update_inflight_metrics()
        return

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

        verify_loop_started_at = perf_counter()
        for item in prepared_jobs:
            if item["prepared"].get("status") != "ok":
                continue

            job_id = item["job_id"]
            payload = item["payload"]
            original_image_bytes = item["original_image_bytes"]
            image_hash = payload.get("image_hash")
            created_at = item["created_at"]
            job_started_at = item["job_started_at"]
            prepared = item["prepared"]
            prepared_timings = prepared.get("timings", {})
            prepared_timings["batch_verify_loop_wait_ms"] = (
                perf_counter() - verify_loop_started_at
            ) * 1000.0
            now = time.time()
            age_ms = (now - created_at) * 1000.0

            if age_ms > MAX_JOB_AGE_MS:
                dequeued_at = item.get("dequeued_at", job_started_at)
                VERIFY_REJECTED_JOBS.labels(reason="expired").inc()
                write_metrics = _build_metrics(created_at, job_started_at, time.time(), dequeued_at=dequeued_at)
                _timed_result_write(VerifyResultStore.set_expired, job_id, write_metrics)
                finished_at = time.time()
                metrics = _build_metrics(created_at, job_started_at, finished_at, dequeued_at=dequeued_at)
                _observe_async_job_metrics(metrics, expired=True)
                JOB_AGE_MS.observe((finished_at - created_at) * 1000.0)
                _print_metrics(job_id, metrics)
                inflight_decrements += 1
                continue

            try:
                t_verify = perf_counter()
                result = await service.verify_from_pipeline_result(
                    prepared,
                    image_bytes=original_image_bytes,
                    user_id=payload.get("user_id"),
                    require_liveness=payload.get("require_liveness", False),
                    check_replay=True,
                    job_id=job_id,
                    t_start=batch_started_at,
                    top_k=item.get("top_k"),
                    image_hash=image_hash,
                )
                prepared_timings["verify_from_pipeline_result_ms"] = (
                    perf_counter() - t_verify
                ) * 1000.0
                _observe_worker_stage(
                    "verify_from_pipeline_result",
                    prepared_timings["verify_from_pipeline_result_ms"],
                )
                service_timings = result.get("timings", {})
                if service_timings:
                    prepared_timings.update(service_timings)
                _observe_prepared_timings(prepared_timings)
                result["bbox_source"] = prepared.get("bbox_source")
                result["bbox_source_detail"] = prepared.get("bbox_source_detail")

                dequeued_at = item.get("dequeued_at", job_started_at)
                print(
                    "[PIPELINE] "
                    f"job={job_id} "
                    f"pre={prepared_timings.get('preprocess_ms', 0):.2f} "
                    f"qpre={prepared_timings.get('quality_gate_pre_ms', 0):.2f} "
                    f"detect={prepared_timings.get('detect_ms', 0):.2f} "
                    f"crop={prepared_timings.get('align_crop_ms', 0):.2f} "
                    f"qface={prepared_timings.get('quality_gate_face_ms', 0):.2f} "
                    f"live={prepared_timings.get('liveness_ms', 0):.2f} "
                    f"vec={prepared_timings.get('vector_search_ms', 0):.2f}"
                )
                write_metrics = _build_metrics(created_at, job_started_at, time.time(), dequeued_at=dequeued_at)
                result_write_ms = _timed_result_write(VerifyResultStore.set_done, job_id, result, write_metrics)
                finished_at = time.time()
                metrics = _build_metrics(created_at, job_started_at, finished_at, dequeued_at=dequeued_at)
                _observe_async_job_metrics(metrics, completed=True)
                JOB_AGE_MS.observe((finished_at - created_at) * 1000.0)
                _print_metrics(job_id, metrics)
                metrics["result_write_ms"] = result_write_ms
                _log_stage_times(
                    job_id=job_id,
                    created_at=created_at,
                    job_started_at=job_started_at,
                    finished_at=finished_at,
                    prepared_timings=prepared_timings,
                    result_write_ms=metrics["result_write_ms"],
                    vector_search_ms=prepared_timings.get("vector_search_ms"),
                    dequeued_at=item.get("dequeued_at", job_started_at),
                    outcome="ok",
                    quality_reason=prepared.get("quality_reason"),
                    quality_details=prepared.get("quality_details", {}),
                )
                inflight_decrements += 1
            except Exception as exc:
                dequeued_at = item.get("dequeued_at", job_started_at)
                write_metrics = _build_metrics(created_at, job_started_at, time.time(), dequeued_at=dequeued_at)
                result_write_ms = _timed_result_write(VerifyResultStore.set_error, job_id, str(exc), write_metrics)
                finished_at = time.time()
                metrics = _build_metrics(created_at, job_started_at, finished_at, dequeued_at=dequeued_at)
                _observe_async_job_metrics(metrics, completed=True)
                JOB_AGE_MS.observe((finished_at - created_at) * 1000.0)
                _print_metrics(job_id, metrics)
                metrics["result_write_ms"] = result_write_ms
                _log_stage_times(
                    job_id=job_id,
                    created_at=created_at,
                    job_started_at=job_started_at,
                    finished_at=finished_at,
                    prepared_timings=prepared_timings,
                    result_write_ms=metrics["result_write_ms"],
                    vector_search_ms=prepared_timings.get("vector_search_ms"),
                    dequeued_at=dequeued_at,
                    outcome="error",
                    quality_reason=prepared.get("quality_reason"),
                    quality_details=prepared.get("quality_details", {}),
                )
                inflight_decrements += 1

        t_commit = perf_counter()
        await db.commit()
        batch_db_commit_wall_ms = (perf_counter() - t_commit) * 1000.0
        logger.info("[BATCH COMMIT] commit_ms=%.3f size=%s", batch_db_commit_wall_ms, len(ok_jobs))

    if inflight_decrements:
        _decr_inflight(inflight_decrements)
        _update_inflight_metrics()


async def run_worker():
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    start_http_server(9101)
    await warmup()

    while True:
        batch = await collect_batch()
        if batch:
            await process_batch(batch)


if __name__ == "__main__":
    setup_logging()
    asyncio.run(run_worker())
