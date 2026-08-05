# app/workers/verify_worker.py

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

from app.core.timing import StageTimings, now_epoch_ns
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.verification_repo import VerificationRepository
from app.core.config import settings
from app.core.logger import setup_logging
from app.core.request_context import reset_client_ip, set_client_ip
from app.db.session import AsyncSessionLocal
from app.infrastructure.minio_client import MinioClient
from app.monitoring.metrics import (
    ASYNC_JOB_COMPLETED_TOTAL,
    ASYNC_JOB_EXPIRED_TOTAL,
    ASYNC_JOB_E2E_LATENCY_MS,
    ASYNC_JOB_PROCESSING_MS,
    ASYNC_JOB_QUEUE_DELAY_MS,
    ASYNC_JOB_TOTAL_LATENCY_MS,
    VERIFY_JOB_AGE_ON_FINALIZE_MS,
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
    QUEUE_BATCH_SIZE,
    QUEUE_JOBS_PER_POP,
    QUEUE_ASSIGNMENT_DELAY_MS,
    WORKER_IDLE_GAP_MS,
    WORKER_ACTIVE_BATCHES,
    WORKER_SEMAPHORE_WAIT_MS,
    QUEUE_CLAIM_ATTEMPTS_TOTAL,
    QUEUE_CLAIM_SUCCESS_TOTAL,
    QUEUE_TIME_TO_FIRST_CLAIM_MS,
    QUEUE_CLAIM_TO_BATCH_FILL_MS,
    QUEUE_BATCH_READY_TO_PROCESSING_START_MS,
    QUEUE_ENQUEUE_TO_WORKER_ATTEMPT_MS,
    QUEUE_WORKER_ATTEMPT_TO_CLAIM_SUCCESS_MS,
    VECTOR_SEARCH_MS,
    VERIFY_REJECTED_JOBS,
    QUEUE_LENGTH_REDIS_SNAPSHOT,
    MINIO_DELETE_FAIL_TOTAL,
    inc_async_stage_failure,
    observe_async_stage,
    observe_pipeline_stage,
    VERIFY_JOB_TERMINAL_TOTAL,
    VERIFY_WORKER_CLAIM_TO_FINALIZE_MS,
    VERIFY_WORKER_CLAIM_TO_RESULT_VISIBLE_MS,
    VERIFY_WORKER_FINALIZE_FAIL_TOTAL,
    VERIFY_WORKER_FINALIZE_TOTAL,
    VERIFY_WORKER_RESULT_WRITE_MS,
    VERIFY_WORKER_TERMINAL_GAP_MS,
)
from prometheus_client import start_http_server
from app.services.search_service import SearchService
from app.services.verify_job_queue import VerifyJobQueue
from app.services.verify_result_store import VerifyResultStore
from app.services.verification_service import VerificationService
from app.services.webhook_service import notify_sync as _webhook_notify_sync

logger = logging.getLogger(__name__)
METRICS_ENABLED = True


def _job_extra(job_data: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    """Сформировать extra-поля для лога worker'а с trace_id из job payload."""
    extra = dict(kwargs)
    trace_id = _trace_id_from_job_data(job_data)
    if trace_id is not None:
        extra["trace_id"] = trace_id
    job_id = job_data.get("job_id") if job_data is not None else None
    if job_id is not None:
        extra["job_id"] = job_id
    return extra


def _trace_id_from_job_data(job_data: dict[str, Any] | None) -> str | None:
    """Извлечь trace_id из job payload (если есть)."""
    if job_data is None:
        return None
    payload = job_data.get("payload") or {}
    return payload.get("trace_id")


def _dispatch_webhook(job_id: str, terminal_state: str) -> None:
    """
    Fire-and-forget webhook-уведомление о терминальном состоянии job'а (ТЗ 3.2).
    Перечитывает уже sanitised-конверт из Redis (job:{job_id} — записан moments earlier
    через VerifyResultStore.set_done/set_error/set_expired), поэтому image_b64 и
    прочие _SENSITIVE_KEYS в payload не попадают. expired не отправляем
    (клиент, вероятнее всего, уже отказался от результата).
    """
    if not settings.WEBHOOK_ENABLED:
        return
    if terminal_state == "expired":
        return
    try:
        envelope = VerifyResultStore.get(job_id)
    except Exception:
        logger.warning("webhook_envelope_read_failed job_id=%s", job_id, exc_info=True)
        envelope = None
    payload = envelope if envelope is not None else {"status": terminal_state}
    _webhook_notify_sync(job_id, terminal_state, payload)

redis_client = redis.from_url(settings.redis_url, decode_responses=True)
QUEUE_NAME = "face_verify_queue"
MAX_QUEUE_WAIT_SEC = float(os.getenv("MAX_QUEUE_WAIT_SEC", "15.0"))
MAX_JOB_AGE_MS = int(os.getenv("MAX_JOB_AGE_MS", "0"))
ENABLE_WORKER_EXPIRY = os.getenv("ENABLE_WORKER_EXPIRY", "false").lower() == "true"
BATCH_COLLECT_TIMEOUT = float(os.getenv("BATCH_COLLECT_TIMEOUT", "0.005"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
_PIPELINE: Any | None = None

ENCODE_CONCURRENCY = max(1, int(settings.WORKER_SEMAPHORE))
MAX_ACTIVE_BATCH_TASKS = max(
    1,
    int(os.getenv("WORKER_MAX_ACTIVE_BATCHES", str(ENCODE_CONCURRENCY))),
)

# This semaphore only gates the encode stage.
encode_semaphore = asyncio.Semaphore(ENCODE_CONCURRENCY)


def _log_service_runtime_snapshot() -> None:
    logger.info(
        "service_runtime_snapshot",
        extra=settings.service_runtime_snapshot(
            "worker",
            worker_batch_size=BATCH_SIZE,
            worker_batch_collect_timeout_ms=int(BATCH_COLLECT_TIMEOUT * 1000.0),
        ),
    )


def _decode_job_payload_sync(
    image_url: str,
) -> tuple[bytes, np.ndarray | None, dict[str, float]]:
    """Скачать image_url из MinIO → декодировать → downscale.

    Plaintext base64 НЕ лежит в Redis-очереди (152-ФЗ): route загружает фото в
    MinIO и передаёт только object_name. Байты скачиваются здесь и живут только
    в памяти воркера до завершения обработки.
    """
    t0 = perf_counter()
    minio_client = MinioClient()
    original_image_bytes = minio_client.get_image(image_url)
    minio_download_ms = (perf_counter() - t0) * 1000.0

    image, worker_pre_timings = _decode_image(original_image_bytes)
    worker_pre_timings["minio_download_ms"] = minio_download_ms
    return original_image_bytes, image, worker_pre_timings


def _cleanup_minio_image(image_url: str | None, job_id: str, stage: str = "worker") -> None:
    """Best-effort удаление исходного фото из MinIO после обработки воркером.

    Молчит при отсутствии image_url (legacy/битый payload) и при ошибке удаления
    (MinIO lifecycle-cover — резервная очистка). Отказы удаления инкрементим
    метрикой MINIO_DELETE_FAIL_TOTAL (stage-метка) для observability — раньше
    это делал только legacy Celery verify_task (удалён), теперь единый путь здесь.
    """
    if not image_url:
        return
    try:
        MinioClient().delete_image(image_url)
    except Exception:
        MINIO_DELETE_FAIL_TOTAL.labels(stage=stage).inc()
        logger.warning(
            "minio_delete_failed job_id=%s image_url=%s stage=%s",
            job_id, image_url, stage,
            exc_info=True,
        )


def _prepare_face_inputs_sync(
    pipeline: Any,
    batch_candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    t0 = perf_counter()
    decoded_images = [item.get("image") for item in batch_candidates]
    if all(image is not None for image in decoded_images):
        prepared_results = pipeline.prepare_face_inputs_from_images(
            [cast(np.ndarray, image) for image in decoded_images]
        )
    else:
        image_bytes_list = [item["image_bytes"] for item in batch_candidates]
        prepared_results = pipeline.prepare_face_inputs(image_bytes_list)

    prep_ms = (perf_counter() - t0) * 1000.0
    detector = getattr(pipeline, "fast_detector", None)
    detector_batch_timings = getattr(detector, "last_batch_timings", {}) or {}
    return prepared_results, prep_ms, detector_batch_timings


def _encode_batch_sync(
    pipeline: Any,
    face_inputs: list[Any],
) -> tuple[list[Any], float, dict[str, Any]]:
    t0 = perf_counter()
    embeddings = pipeline.encoder.encode_batch(face_inputs)
    batch_encode_ms = (perf_counter() - t0) * 1000.0

    encoder_wrapper = getattr(pipeline, "encoder", None)
    encoder_impl = getattr(encoder_wrapper, "encoder", encoder_wrapper)
    encoder_batch_timings = getattr(encoder_impl, "last_batch_timings", {}) or {}
    return embeddings, batch_encode_ms, encoder_batch_timings


def _is_job_stale(created_at: float, observed_at: float) -> bool:
    if not ENABLE_WORKER_EXPIRY:
        return False

    queue_wait_sec = max(0.0, observed_at - created_at)
    queue_wait_ms = queue_wait_sec * 1000.0

    if MAX_QUEUE_WAIT_SEC > 0 and queue_wait_sec > MAX_QUEUE_WAIT_SEC:
        return True
    if MAX_JOB_AGE_MS > 0 and queue_wait_ms > float(MAX_JOB_AGE_MS):
        return True
    return False


async def _redis_llen(metric_command: str = "llen_queue") -> int:
    start = perf_counter()
    raw = await asyncio.to_thread(redis_client.llen, QUEUE_NAME)
    REDIS_COMMAND_LATENCY_MS.labels(command=metric_command).observe(
        (perf_counter() - start) * 1000.0
    )
    return int(cast(int, raw))


async def _update_queue_length_metric_async() -> None:
    queue_length = float(await _redis_llen("llen_queue"))
    QUEUE_LENGTH.set(queue_length)
    QUEUE_LENGTH_REDIS_SNAPSHOT.set(queue_length)


async def _redis_brpop() -> tuple[str, str] | None:
    start = perf_counter()
    raw = await asyncio.to_thread(redis_client.brpop, [QUEUE_NAME], 0)
    QUEUE_POP_LATENCY_MS.observe((perf_counter() - start) * 1000.0)
    return cast(tuple[str, str] | None, raw)


async def _redis_lpop() -> str | None:
    start = perf_counter()
    raw = await asyncio.to_thread(redis_client.lpop, QUEUE_NAME)
    QUEUE_POP_LATENCY_MS.observe((perf_counter() - start) * 1000.0)
    return cast(str | None, raw)


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


def _record_worker_job_timings(timings: StageTimings) -> None:
    for stage, value_ms in timings.values.items():
        if value_ms <= 0:
            continue
        if stage == "pipeline_total_ms":
            observe_pipeline_stage("pipeline_total", value_ms)
        else:
            observe_async_stage(stage.replace("_ms", ""), value_ms)


def _build_result_payload(
    result: dict[str, Any],
    timings: StageTimings,
    timestamps: dict[str, int],
) -> dict[str, Any]:
    payload = dict(result)
    payload["timings"] = {
        **result.get("timings", {}),
        **timings.values,
    }
    payload["timestamps"] = timestamps
    return payload


def _classify_prepare_exception(exc: BaseException) -> dict[str, Any] | None:
    """Маппинг ValueError из _prepare_face_from_detection в осмысленный терминальный
    статус (no_face/quality_reject/processing_failed).

    Эти исключения — НЕ серверный сбой, а ожидаемое условие «нет годного лица»:
      • "Face not detected" / "no face"      → no_face
      • "Multiple faces not allowed"        → no_face (reason=multiple_faces)
      • "bad crop" / "Empty face crop"      → quality_reject (reason=bad_crop)
      • "Low confidence face detection"    → quality_reject (reason=low_confidence)
    Прочие ValueError → processing_failed/invalid_image.
    Не-ValueError → None (настоящий сбой → set_error, _sanitize).

    Возвращает {result, outcome, terminal_state} или None. result НЕ содержит
    ключ "error" (используем reason/error_code) → _sanitize_mapping его не вырежет
    → клиент получает нормальный status+reason для UI/retry (а не opaque "error").
    terminal_state влияет только на метрики/finalize; envelope-статус для set_done
    всегда "done", поэтому клиент видит result.status.
    """
    if not isinstance(exc, ValueError):
        return None
    msg = str(exc).lower()
    if "no face" in msg or "not detected" in msg:
        status, reason, outcome = "no_face", "no_face", "no_face"
    elif "multiple" in msg:
        status, reason, outcome = "no_face", "multiple_faces", "no_face"
    elif "bad crop" in msg or "empty face crop" in msg:
        status, reason, outcome = "quality_reject", "bad_crop", "quality_reject"
    elif "low confidence" in msg:
        status, reason, outcome = "quality_reject", "low_confidence", "quality_reject"
    else:
        status, reason, outcome = "processing_failed", "invalid_image", "processing_failed"
    terminal_state = "reject" if status in {"no_face", "quality_reject"} else "error"
    result = {
        "status": status,
        "reason": reason,
        "error_code": reason,
        "liveness_passed": False,
        "replay_detected": False,
        "match_score": None,
        "confidence": None,
    }
    return {"result": result, "outcome": outcome, "terminal_state": terminal_state}


def _build_technical_timestamps(
    *,
    queue_popped_at: float | None = None,
    worker_started_at_ns: int | None = None,
    completed_at_ns: int | None = None,
    result_visible_at_ms: int | None = None,
) -> dict[str, int]:
    technical: dict[str, int] = {}
    if queue_popped_at is not None:
        technical["queue_popped_at_ms"] = int(queue_popped_at * 1000.0)
    if worker_started_at_ns is not None:
        technical["worker_started_at_ms"] = int(worker_started_at_ns / 1_000_000)
    if completed_at_ns is not None:
        technical["completed_at_ms"] = int(completed_at_ns / 1_000_000)
    if result_visible_at_ms is not None:
        technical["result_visible_at_ms"] = int(result_visible_at_ms)
    return technical


def _payload_timestamp_ns(payload: dict[str, Any], key: str, fallback_ns: int) -> int:
    raw = payload.get(key)
    if raw is None:
        return fallback_ns
    try:
        return int(raw)
    except (TypeError, ValueError):
        return fallback_ns


PIPELINE_TOTAL_STAGE_KEYS = (
    "preprocess_ms",
    "detect_ms",
    "align_ms",
    "encode_ms",
    "search_ms",
    "liveness_ms",
    "decision_ms",
)


def _normalize_job_stage_timings(prepared_timings: dict[str, Any] | None) -> dict[str, float]:
    timings = dict(prepared_timings or {})

    timings["preprocess_ms"] = float(timings.get("preprocess_ms", 0.0) or 0.0)
    timings["detect_ms"] = float(
        timings.get("detect_ms", timings.get("fast_detect_ms", 0.0)) or 0.0
    )
    timings["align_ms"] = float(timings.get("align_ms", timings.get("align_crop_ms", 0.0)) or 0.0)
    timings["encode_ms"] = float(timings.get("encode_ms", 0.0) or 0.0)
    timings["search_ms"] = float(timings.get("search_ms", timings.get("vector_search_ms", 0.0)) or 0.0)
    timings["liveness_ms"] = float(timings.get("liveness_ms", 0.0) or 0.0)
    timings["decision_ms"] = float(timings.get("decision_ms", 0.0) or 0.0)
    timings["pipeline_total_ms"] = sum(
        float(timings.get(stage, 0.0) or 0.0) for stage in PIPELINE_TOTAL_STAGE_KEYS
    )
    return timings


def _extract_stage_timings(prepared_timings: dict[str, Any] | None) -> dict[str, float]:
    timings = _normalize_job_stage_timings(prepared_timings)

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
        "enqueue_to_worker_attempt_ms": float(timings.get("enqueue_to_worker_attempt_ms", 0.0)),
        "worker_attempt_to_claim_success_ms": float(
            timings.get("worker_attempt_to_claim_success_ms", 0.0)
        ),
        "time_to_first_claim_ms": float(timings.get("time_to_first_claim_ms", 0.0)),
        "claim_to_batch_fill_ms": float(timings.get("claim_to_batch_fill_ms", 0.0)),
        "batch_ready_to_processing_start_ms": float(
            timings.get("batch_ready_to_processing_start_ms", 0.0)
        ),
        "preprocess_ms": float(timings.get("preprocess_ms", 0.0)),
        "quality_gate_pre_ms": float(timings.get("quality_gate_pre_ms", 0.0)),
        "worker_semaphore_wait_ms": float(timings.get("worker_semaphore_wait_ms", 0.0)),
        "encode_semaphore_wait_ms": float(timings.get("encode_semaphore_wait_ms", 0.0)),
        "detect_ms": float(timings.get("detect_ms", 0.0)),
        "detect_blob_ms": float(timings.get("detect_blob_ms", 0.0)),
        "detect_forward_ms": float(timings.get("detect_forward_ms", 0.0)),
        "detect_decode_ms": float(timings.get("detect_decode_ms", 0.0)),
        "align_crop_ms": float(timings.get("align_crop_ms", 0.0)),
        "align_ms": float(timings.get("align_ms", 0.0)),
        "quality_gate_face_ms": float(timings.get("quality_gate_face_ms", 0.0)),
        "liveness_ms": float(timings.get("liveness_ms", 0.0)),
        "encode_ms": float(timings.get("encode_ms", 0.0)),
        "encode_preprocess_ms": float(timings.get("encode_preprocess_ms", 0.0)),
        "encode_ort_run_ms": float(timings.get("encode_ort_run_ms", 0.0)),
        "encode_postprocess_ms": float(timings.get("encode_postprocess_ms", 0.0)),
        "search_ms": float(timings.get("search_ms", 0.0)),
        "vector_search_ms": float(timings.get("vector_search_ms", 0.0)),
        "anti_replay_ms": float(timings.get("anti_replay_ms", 0.0)),
        "is_genuine_ms": float(timings.get("is_genuine_ms", 0.0)),
        "decision_ms": float(timings.get("decision_ms", 0.0)),
        "pipeline_total_ms": float(timings.get("pipeline_total_ms", 0.0)),
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
            "queue_wait_ms",
            "batch_wait_ms",
            "worker_semaphore_wait_ms",
            "encode_semaphore_wait_ms",
            "pipeline_total_ms",
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
    trace_id: str | None = None,
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
        + stage_timings["encode_semaphore_wait_ms"]
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

    extra: dict[str, Any] = {
        "job_id": job_id,
        "outcome": outcome or "unknown",
        "quality_reason": quality_reason,
        "quality_stage": quality_stage,
        "quality_mode": quality_mode,
        "quality_warning": quality_warning,
        "image_width": image_width,
        "image_height": image_height,
        "blur_score": blur_score,
        "brightness": brightness,
        "contrast": contrast,
        "face_width": face_width,
        "face_height": face_height,
        "min_face_side": min_face_side,
        "queue_delay_ms": queue_delay_ms,
        "dequeue_to_start_ms": dequeue_to_start_ms,
        "processing_ms": processing_ms,
        "e2e_ms": e2e_ms,
        "unattributed_ms": unattributed_ms,
        "result_write_ms": result_write_value,
        "faiss_enabled": bool(settings.FAISS_ENABLED),
    }
    # trace_id передаём только если есть — для корреляции с API-логами.
    if trace_id is not None:
        extra["trace_id"] = trace_id
    # Все стадийные тайминги — без биометрии, безопасно для логов.
    extra.update(stage_timings)
    logger.info("stage_times", extra=extra)


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
    logger.info(
        "job_metrics",
        extra={
            "job_id": job_id,
            "queue_delay_s": metrics.get("queue_delay"),
            "dequeue_to_start_s": metrics.get("dequeue_to_start"),
            "processing_time_s": metrics.get("processing_time"),
            "total_latency_s": metrics.get("total_latency"),
        },
    )


def _finalize_terminal_job(
    *,
    job_id: str,
    terminal_state: str,
    created_at: float,
    dequeued_at: float,
    result_write_ms: float,
    result_written_at: float | None,
    job_timings: StageTimings | None = None,
    prepared_timings: dict[str, Any] | None = None,
    vector_search_ms: float | None = None,
    claim_at: float | None = None,
    quality_reason: str | None = None,
    quality_details: dict[str, Any] | None = None,
) -> None:
    finalize_start = time.time()
    claim_anchor = claim_at if claim_at is not None else dequeued_at
    visible_at = result_written_at if result_written_at is not None else finalize_start
    try:
        VERIFY_JOB_TERMINAL_TOTAL.labels(state=terminal_state).inc()
        VERIFY_WORKER_RESULT_WRITE_MS.observe(max(0.0, float(result_write_ms)))
        VERIFY_WORKER_CLAIM_TO_RESULT_VISIBLE_MS.observe(
            max(0.0, (visible_at - claim_anchor) * 1000.0)
        )
        VERIFY_WORKER_CLAIM_TO_FINALIZE_MS.observe(
            max(0.0, (finalize_start - claim_anchor) * 1000.0)
        )
        VERIFY_WORKER_TERMINAL_GAP_MS.observe(
            max(0.0, (finalize_start - visible_at) * 1000.0)
        )
        VERIFY_JOB_AGE_ON_FINALIZE_MS.observe((finalize_start - created_at) * 1000.0)
        VerifyJobQueue.finalize_job(job_id=job_id, terminal_state=terminal_state)
        VERIFY_WORKER_FINALIZE_TOTAL.labels(state=terminal_state).inc()
    except Exception:
        VERIFY_WORKER_FINALIZE_FAIL_TOTAL.inc()
        logger.exception("worker_finalize_failed job_id=%s state=%s", job_id, terminal_state)

    # Webhook: fire-and-forget, не должен влиять на метрики finalize.
    try:
        _dispatch_webhook(job_id, terminal_state)
    except Exception:
        logger.warning("webhook_dispatch_failed job_id=%s state=%s", job_id, terminal_state, exc_info=True)


def _complete_terminal_job_inline(
    *,
    job_id: str,
    terminal_state: str,
    created_at: float,
    job_started_at: float,
    dequeued_at: float,
    result_write_ms: float,
    claim_at: float | None = None,
    job_timings: StageTimings | None = None,
    prepared_timings: dict[str, Any] | None = None,
    vector_search_ms: float | None = None,
    outcome: str | None = None,
    quality_reason: str | None = None,
    quality_details: dict[str, Any] | None = None,
    trace_id: str | None = None,
) -> None:
    result_written_at = time.time()

    _safe_observe_terminal_job(
        job_id=job_id,
        created_at=created_at,
        job_started_at=job_started_at,
        dequeued_at=dequeued_at,
        result_write_ms=result_write_ms,
        completed=True,
        expired=(terminal_state == "expired"),
        job_timings=job_timings,
        prepared_timings=prepared_timings,
        vector_search_ms=vector_search_ms,
        outcome=outcome,
        quality_reason=quality_reason,
        quality_details=quality_details,
        trace_id=trace_id,
    )

    _finalize_terminal_job(
        job_id=job_id,
        terminal_state=terminal_state,
        created_at=created_at,
        dequeued_at=dequeued_at,
        result_write_ms=result_write_ms,
        result_written_at=result_written_at,
        job_timings=job_timings,
        prepared_timings=prepared_timings,
        vector_search_ms=vector_search_ms,
        claim_at=claim_at,
        quality_reason=quality_reason,
        quality_details=quality_details,
    )


def _safe_observe_terminal_job(
    *,
    job_id: str,
    created_at: float,
    job_started_at: float,
    dequeued_at: float,
    result_write_ms: float,
    completed: bool = False,
    expired: bool = False,
    job_timings: StageTimings | None = None,
    prepared_timings: dict[str, Any] | None = None,
    vector_search_ms: float | None = None,
    outcome: str | None = None,
    quality_reason: str | None = None,
    quality_details: dict[str, Any] | None = None,
    trace_id: str | None = None,
) -> None:
    try:
        finished_at = time.time()
        metrics = _build_metrics(
            created_at,
            job_started_at,
            finished_at,
            dequeued_at=dequeued_at,
        )
        _observe_async_job_metrics(metrics, completed=completed, expired=expired)
        if job_timings is not None:
            _record_worker_job_timings(job_timings)
        JOB_AGE_MS.observe((finished_at - created_at) * 1000.0)
        _print_metrics(job_id, metrics)
        _log_stage_times(
            job_id=job_id,
            created_at=created_at,
            job_started_at=job_started_at,
            finished_at=finished_at,
            prepared_timings=prepared_timings or {},
            result_write_ms=result_write_ms,
            vector_search_ms=vector_search_ms,
            dequeued_at=dequeued_at,
            outcome=outcome,
            quality_reason=quality_reason,
            quality_details=quality_details or {},
            trace_id=trace_id,
        )
    except Exception:
        logger.exception(
            "terminal_observe_failed",
            extra={"job_id": job_id, "outcome": outcome or "unknown", "trace_id": trace_id},
        )


def _reject_stale_job(job_id: str, created_at: float, observed_at: float, trace_id: str | None = None) -> None:
    write_metrics = _build_metrics(created_at, observed_at, time.time(), dequeued_at=observed_at)
    technical_timestamps = _build_technical_timestamps(
        queue_popped_at=observed_at,
        worker_started_at_ns=int(observed_at * 1_000_000_000),
        completed_at_ns=int(time.time() * 1_000_000_000),
        result_visible_at_ms=int(time.time() * 1000.0),
    )
    result_write_ms = _timed_result_write(
        VerifyResultStore.set_expired,
        job_id,
        write_metrics,
        technical_timestamps=technical_timestamps,
    )
    VERIFY_REJECTED_JOBS.labels(reason="expired").inc()
    _complete_terminal_job_inline(
        job_id=job_id,
        terminal_state="expired",
        created_at=created_at,
        job_started_at=observed_at,
        dequeued_at=observed_at,
        result_write_ms=result_write_ms,
        claim_at=observed_at,
        outcome="expired",
        trace_id=trace_id,
    )


def _decode_image(
    image_bytes: bytes,
) -> tuple[np.ndarray | None, dict[str, float]]:
    """Декодировать bytes в ndarray (full-res, БЕЗ downscale).

    Downscale больше не делается на стороне воркера: pipeline хранит original
    (кроп лица/occ/embedding/liveness берутся из full-res, чтобы не терять разрешение
    на 16-МП фото) и отдельно даунскейлит только кадр для быстрой детекции
    (decode_pair / process_image внутри pipeline). Ключи downscale_ms/jpeg_reencode_ms
    сохранены как 0.0 — их читают метрики/логи worker-pre stage.
    """
    timings: dict[str, float] = {}

    t0 = perf_counter()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    timings["image_decode_ms"] = (perf_counter() - t0) * 1000.0

    timings["downscale_ms"] = 0.0
    timings["jpeg_reencode_ms"] = 0.0

    if decoded is None:
        return None, timings

    return decoded, timings


async def warmup():
    global _PIPELINE

    logger.info("pipeline_warmup_started")

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

    logger.info("pipeline_warmup_done")


async def collect_batch() -> list[dict[str, Any]]:
    await _redis_llen("llen_queue_pre_batch")
    await _update_queue_length_metric_async()

    jobs: list[dict[str, Any]] = []
    batch_size = BATCH_SIZE

    first_job: dict[str, Any] | None = None
    first_claim_at: float | None = None
    first_worker_attempt_at: float | None = None
    first_claim_success_at: float | None = None
    while first_job is None:
        QUEUE_CLAIM_ATTEMPTS_TOTAL.inc()
        attempt_started_at = time.time()
        start = perf_counter()
        raw = await _redis_brpop()
        if raw is None:
            return []

        QUEUE_CLAIM_SUCCESS_TOTAL.inc()
        _, data = raw
        idle_gap_ms = (perf_counter() - start) * 1000.0
        claim_at = time.time()
        job = json.loads(data)
        job["dequeued_at"] = claim_at
        job["batch_added_at"] = claim_at
        created_at = float(job.get("created_at", claim_at))
        if _is_job_stale(created_at, claim_at):
            _reject_stale_job(
                job["job_id"],
                created_at,
                claim_at,
                trace_id=_trace_id_from_job_data(job),
            )
            _cleanup_minio_image(
                job.get("payload", {}).get("image_url"),
                job["job_id"],
                stage="stale",
            )
            continue

        WORKER_IDLE_GAP_MS.observe(idle_gap_ms)
        first_worker_attempt_at = attempt_started_at
        first_claim_success_at = claim_at
        first_claim_at = claim_at
        first_claim_at_ns = now_epoch_ns()
        first_job = job
        job["worker_claimed_at_ns"] = first_claim_at_ns
        jobs.append(job)

    while len(jobs) < batch_size:
        QUEUE_CLAIM_ATTEMPTS_TOTAL.inc()
        raw = await _redis_lpop()
        if raw is None:
            break

        QUEUE_CLAIM_SUCCESS_TOTAL.inc()
        claim_at = time.time()
        claim_at_ns = now_epoch_ns()
        job = json.loads(raw)
        job["dequeued_at"] = claim_at
        job["batch_added_at"] = claim_at
        job["worker_claimed_at_ns"] = claim_at_ns
        created_at = float(job.get("created_at", claim_at))
        if _is_job_stale(created_at, claim_at):
            _reject_stale_job(
                job["job_id"],
                created_at,
                claim_at,
                trace_id=_trace_id_from_job_data(job),
            )
            _cleanup_minio_image(
                job.get("payload", {}).get("image_url"),
                job["job_id"],
                stage="stale",
            )
            continue

        jobs.append(job)

    await _update_queue_length_metric_async()
    batch_collected_at = time.time()
    if first_job is not None:
        first_job_created_at = float(first_job.get("created_at", batch_collected_at))
        enqueue_to_worker_attempt_ms = max(
            0.0,
            (float(first_worker_attempt_at or batch_collected_at) - first_job_created_at) * 1000.0,
        )
        worker_attempt_to_claim_success_ms = max(
            0.0,
            (float(first_claim_success_at or batch_collected_at) - float(first_worker_attempt_at or batch_collected_at))
            * 1000.0,
        )
        QUEUE_TIME_TO_FIRST_CLAIM_MS.observe(
            max(0.0, (float(first_claim_at or batch_collected_at) - first_job_created_at) * 1000.0)
        )
        QUEUE_CLAIM_TO_BATCH_FILL_MS.observe(
            max(0.0, (batch_collected_at - float(first_claim_at or batch_collected_at)) * 1000.0)
        )
        QUEUE_ENQUEUE_TO_WORKER_ATTEMPT_MS.observe(enqueue_to_worker_attempt_ms)
        QUEUE_WORKER_ATTEMPT_TO_CLAIM_SUCCESS_MS.observe(worker_attempt_to_claim_success_ms)
    for job in jobs:
        job["batch_collected_at"] = batch_collected_at
        if first_claim_at is not None:
            job["first_claim_at"] = first_claim_at
        if first_worker_attempt_at is not None:
            job["worker_attempt_started_at"] = first_worker_attempt_at
        if first_claim_success_at is not None:
            job["worker_claim_success_at"] = first_claim_success_at
    batch_collected_at_ns = now_epoch_ns()
    for job in jobs:
        job["batch_collected_at_ns"] = batch_collected_at_ns
    return jobs


async def process_batch(job_datas: list[dict[str, Any]]):
    pipeline = _PIPELINE
    if pipeline is None:
        raise RuntimeError("Pipeline is required for batch processing")

    batch_started_at = time.time()
    if job_datas:
        QUEUE_BATCH_SIZE.observe(float(len(job_datas)))
        QUEUE_JOBS_PER_POP.observe(float(len(job_datas)))
        batch_collected_at = float(job_datas[0].get("batch_collected_at", batch_started_at))
        QUEUE_BATCH_READY_TO_PROCESSING_START_MS.observe(
            max(0.0, (batch_started_at - batch_collected_at) * 1000.0)
        )
    WORKER_ACTIVE_BATCHES.inc()
    prepared_jobs: list[dict[str, Any]] = []
    batch_candidates: list[dict[str, Any]] = []
    batch_collected_at = (
        float(job_datas[0].get("batch_collected_at", batch_started_at)) if job_datas else batch_started_at
    )
    try:

        for job_data in job_datas:
            job_started_at = time.time()
            job_id = job_data["job_id"]
            payload = job_data["payload"]
            created_at = job_data.get("created_at", batch_started_at)
            dequeued_at = float(job_data.get("dequeued_at", job_started_at))
            batch_collected_at = float(job_data.get("batch_collected_at", batch_started_at))
            job_timings = StageTimings()
            accepted_at_ns = _payload_timestamp_ns(payload, "accepted_at_ns", int(created_at * 1_000_000_000))
            enqueued_at_ns = _payload_timestamp_ns(payload, "enqueued_at_ns", accepted_at_ns)
            worker_claimed_at_ns = int(job_data.get("worker_claimed_at_ns", now_epoch_ns()))
            batch_collected_at_ns = int(
                job_data.get("batch_collected_at_ns", now_epoch_ns())
            )
            job_timings.set(
                "queue_wait_ms",
                max(0.0, (worker_claimed_at_ns - enqueued_at_ns) / 1_000_000.0),
            )
            job_timings.set(
                "batch_wait_ms",
                max(0.0, (batch_collected_at_ns - worker_claimed_at_ns) / 1_000_000.0),
            )
            QUEUE_ASSIGNMENT_DELAY_MS.observe(max(0.0, (batch_collected_at - created_at) * 1000.0))

            try:
                original_image_bytes, image, worker_pre_timings = await asyncio.to_thread(
                    _decode_job_payload_sync,
                    payload.get("image_url"),
                )
                first_claim_at_for_job = float(job_data.get("first_claim_at", dequeued_at))
                worker_pre_timings["batch_collect_wait_ms"] = max(
                    0.0,
                    (batch_started_at - dequeued_at) * 1000.0,
                )
                worker_pre_timings["time_to_first_claim_ms"] = max(
                    0.0,
                    (first_claim_at_for_job - created_at) * 1000.0,
                )
                worker_pre_timings["claim_to_batch_fill_ms"] = max(
                    0.0,
                    (batch_collected_at - first_claim_at_for_job)
                    * 1000.0,
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
                        "first_claim_at": first_claim_at_for_job,
                        "batch_collected_at": batch_collected_at,
                        "worker_claimed_at_ns": worker_claimed_at_ns,
                        "batch_collected_at_ns": batch_collected_at_ns,
                        "job_timings": job_timings,
                        "accepted_at_ns": accepted_at_ns,
                        "enqueued_at_ns": enqueued_at_ns,
                        "worker_pre_timings": worker_pre_timings,
                    }
                )
            except Exception as exc:
                inc_async_stage_failure("pipeline", exc.__class__.__name__)
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
                _complete_terminal_job_inline(
                    job_id=job_id,
                    terminal_state="error",
                    created_at=created_at,
                    job_started_at=job_started_at,
                    dequeued_at=dequeued_at,
                    result_write_ms=result_write_ms,
                    claim_at=worker_claimed_at_ns / 1_000_000_000.0,
                    job_timings=job_timings,
                    outcome="processing_failed",
                    trace_id=_trace_id_from_job_data(job_data),
                )

        if batch_candidates:
            try:
                prepared_results, prep_ms, detector_batch_timings = await asyncio.to_thread(
                    _prepare_face_inputs_sync,
                    pipeline,
                    batch_candidates,
                )
                logger.info(
                    "batch_prepare_done",
                    extra={
                        "batch_size": len(batch_candidates),
                        "batch_prepare_ms": prep_ms,
                    },
                )

                if len(prepared_results) != len(batch_candidates):
                    raise RuntimeError("Pipeline returned unexpected batch size")

                for item, prepared in zip(batch_candidates, prepared_results):
                    prepared_timings = prepared.setdefault("timings", {})
                    prepared_timings.update(item.get("worker_pre_timings", {}))
                    prepared_timings["batch_prepare_wall_ms"] = prep_ms
                    prepared_timings["batch_ready_to_processing_start_ms"] = max(
                        0.0,
                        (batch_started_at - batch_collected_at) * 1000.0,
                    )

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
                inc_async_stage_failure("pipeline", exc.__class__.__name__)
                # ValueError из _prepare_face_from_detection — это НЕ серверный сбой, а
                # ожидаемое условие «нет годного лица» (Face not detected / Multiple
                # faces / bad crop / Empty face crop / Low confidence). Раньше такие
                # кадры уходили в set_error → _sanitize вырезало reason → клиент видел
                # opaque status="error" без причины. Маппим в осмысленный терминальный
                # статус (no_face/quality_reject) с reason — результат без ключа "error"
                # НЕ санитизируется, клиент получает нормальный статус для UI/retry.
                # Прочие исключения (RuntimeError, неожидаемые) — настоящий сбой → set_error.
                prepared_failure = _classify_prepare_exception(exc)
                if prepared_failure is not None:
                    logger.info(
                        "verify_prepare_rejected batch_size=%s reason=%s msg=%s",
                        len(batch_candidates),
                        prepared_failure["outcome"],
                        exc,
                    )
                else:
                    logger.exception(
                        "verify_prepare_failed batch_size=%s error=%s",
                        len(batch_candidates),
                        exc,
                    )
                for item in batch_candidates:
                    job_id = item["job_id"]
                    created_at = item["created_at"]
                    job_started_at = item["job_started_at"]
                    dequeued_at = item.get("dequeued_at", job_started_at)
                    job_timings = item.get("job_timings")
                    write_metrics = _build_metrics(created_at, job_started_at, time.time(), dequeued_at=dequeued_at)
                    technical_timestamps = _build_technical_timestamps(
                        queue_popped_at=dequeued_at,
                        worker_started_at_ns=int(job_started_at * 1_000_000_000),
                        completed_at_ns=int(time.time() * 1_000_000_000),
                        result_visible_at_ms=int(time.time() * 1000.0),
                    )
                    if prepared_failure is not None:
                        result_write_ms = _timed_result_write(
                            VerifyResultStore.set_done,
                            job_id,
                            dict(prepared_failure["result"]),
                            write_metrics,
                            technical_timestamps=technical_timestamps,
                        )
                        _complete_terminal_job_inline(
                            job_id=job_id,
                            terminal_state=prepared_failure["terminal_state"],
                            created_at=created_at,
                            job_started_at=job_started_at,
                            dequeued_at=dequeued_at,
                            result_write_ms=result_write_ms,
                            claim_at=float(item.get("worker_claimed_at_ns", now_epoch_ns())) / 1_000_000_000.0,
                            job_timings=job_timings,
                            outcome=prepared_failure["outcome"],
                            trace_id=_trace_id_from_job_data(item),
                        )
                    else:
                        result_write_ms = _timed_result_write(
                            VerifyResultStore.set_error,
                            job_id,
                            str(exc),
                            write_metrics,
                            technical_timestamps=technical_timestamps,
                        )
                        _complete_terminal_job_inline(
                            job_id=job_id,
                            terminal_state="error",
                            created_at=created_at,
                            job_started_at=job_started_at,
                            dequeued_at=dequeued_at,
                            result_write_ms=result_write_ms,
                            claim_at=float(item.get("worker_claimed_at_ns", now_epoch_ns())) / 1_000_000_000.0,
                            job_timings=job_timings,
                            outcome="error",
                            trace_id=_trace_id_from_job_data(item),
                        )

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
            job_timings = item.get("job_timings")
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
                _complete_terminal_job_inline(
                    job_id=job_id,
                    terminal_state="reject",
                    created_at=created_at,
                    job_started_at=job_started_at,
                    dequeued_at=dequeued_at,
                    result_write_ms=result_write_ms,
                    claim_at=float(item.get("worker_claimed_at_ns", now_epoch_ns())) / 1_000_000_000.0,
                    job_timings=job_timings,
                    prepared_timings=prepared.get("timings", {}),
                    outcome="quality_reject",
                    quality_reason=prepared.get("quality_reason"),
                    quality_details=prepared.get("quality_details", {}),
                    trace_id=_trace_id_from_job_data(item),
                )
            elif prepared["status"] == "retry":
                # Окклюзия (маска/очки): не исход верификации, а запрос пере-съёмки.
                # Метрику инкрементим как reject-счётчик по reason=remove_occlusion
                # для наблюдаемости; job терминальный (клиент пере-создаёт запрос).
                reason = prepared.get("quality_reason") or "remove_occlusion"
                QUALITY_REJECT_COUNTER.labels(reason=reason).inc()
                write_metrics = _build_metrics(created_at, job_started_at, time.time(), dequeued_at=dequeued_at)
                result_write_ms = _timed_result_write(
                    VerifyResultStore.set_done,
                    job_id,
                    {
                        "status": "retry",
                        "reason": reason,
                        "quality_details": prepared.get("quality_details", {}),
                        "liveness_passed": None,
                        "replay_detected": False,
                        "error_code": reason,
                        "bbox": prepared.get("bbox"),
                        "bbox_source": prepared.get("bbox_source"),
                        "bbox_source_detail": prepared.get("bbox_source_detail"),
                    },
                    write_metrics,
                )
                _complete_terminal_job_inline(
                    job_id=job_id,
                    terminal_state="reject",
                    created_at=created_at,
                    job_started_at=job_started_at,
                    dequeued_at=dequeued_at,
                    result_write_ms=result_write_ms,
                    claim_at=float(item.get("worker_claimed_at_ns", now_epoch_ns())) / 1_000_000_000.0,
                    job_timings=job_timings,
                    prepared_timings=prepared.get("timings", {}),
                    outcome="retry",
                    quality_reason=reason,
                    quality_details=prepared.get("quality_details", {}),
                    trace_id=_trace_id_from_job_data(item),
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
                        "spoofing_indicators": {
                            "real_prob": float(prepared.get("liveness_score", 0.0) or 0.0),
                            "spoof_prob": float(prepared.get("liveness_spoof_score", 0.0) or 0.0),
                        },
                        "replay_detected": False,
                        "error_code": "spoof_detected",
                        "bbox": prepared.get("bbox"),
                        "bbox_source": prepared.get("bbox_source"),
                        "bbox_source_detail": prepared.get("bbox_source_detail"),
                    },
                    write_metrics,
                )
                _complete_terminal_job_inline(
                    job_id=job_id,
                    terminal_state="reject",
                    created_at=created_at,
                    job_started_at=job_started_at,
                    dequeued_at=dequeued_at,
                    result_write_ms=result_write_ms,
                    claim_at=float(item.get("worker_claimed_at_ns", now_epoch_ns())) / 1_000_000_000.0,
                    job_timings=job_timings,
                    prepared_timings=prepared.get("timings", {}),
                    outcome="spoof",
                    quality_reason=prepared.get("quality_reason"),
                    quality_details=prepared.get("quality_details", {}),
                    trace_id=_trace_id_from_job_data(item),
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
                _complete_terminal_job_inline(
                    job_id=job_id,
                    terminal_state="error",
                    created_at=created_at,
                    job_started_at=job_started_at,
                    dequeued_at=dequeued_at,
                    result_write_ms=result_write_ms,
                    claim_at=float(item.get("worker_claimed_at_ns", now_epoch_ns())) / 1_000_000_000.0,
                    job_timings=job_timings,
                    prepared_timings=prepared.get("timings", {}),
                    outcome="processing_failed",
                    quality_reason=prepared.get("quality_reason"),
                    quality_details=prepared.get("quality_details", {}),
                    trace_id=_trace_id_from_job_data(item),
                )

        try:
            if ok_jobs:
                face_inputs = [item["prepared"]["face_input"] for item in ok_jobs]
                t_wait = perf_counter()
                await encode_semaphore.acquire()
                semaphore_wait_ms = (perf_counter() - t_wait) * 1000.0
                WORKER_SEMAPHORE_WAIT_MS.observe(semaphore_wait_ms)
                for item in ok_jobs:
                    prepared_timings = item["prepared"].setdefault("timings", {})
                    prepared_timings["worker_semaphore_wait_ms"] = float(semaphore_wait_ms)
                    prepared_timings["encode_semaphore_wait_ms"] = float(semaphore_wait_ms)
                    job_timings = item.get("job_timings")
                    if job_timings is not None:
                        job_timings.set("worker_semaphore_wait_ms", semaphore_wait_ms)
                        job_timings.set("encode_semaphore_wait_ms", semaphore_wait_ms)
                try:
                    embeddings, batch_encode_ms, encoder_batch_timings = await asyncio.to_thread(
                        _encode_batch_sync,
                        pipeline,
                        face_inputs,
                    )
                finally:
                    encode_semaphore.release()
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

                for item in ok_jobs:
                    job_id = item["job_id"]
                    prepared = item["prepared"]
                    prepared_timings = prepared.get("timings", {})
                    prepared_timings["encode_ms"] = estimated_encode_ms
                    prepared_timings["encode_semaphore_wait_ms"] = float(
                        prepared_timings.get("encode_semaphore_wait_ms", 0.0)
                    )
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
                        logger.info(
                            "encode_stage_timings",
                            extra=_job_extra(
                                item,
                                stage="encode",
                                preprocess_ms=float(prepared_timings.get("preprocess_ms", 0.0)),
                                quality_gate_pre_ms=float(prepared_timings.get("quality_gate_pre_ms", 0.0)),
                                detect_ms=float(prepared_timings.get("detect_ms", 0.0)),
                                detect_forward_ms=float(prepared_timings.get("detect_forward_ms", 0.0)),
                                align_crop_ms=float(prepared_timings.get("align_crop_ms", 0.0)),
                                quality_gate_face_ms=float(prepared_timings.get("quality_gate_face_ms", 0.0)),
                                liveness_ms=float(prepared_timings.get("liveness_ms", 0.0)),
                                encode_ort_run_ms=float(prepared_timings.get("encode_ort_run_ms", 0.0)),
                            ),
                        )

                    logger.info(
                        "encode_done",
                        extra=_job_extra(
                            item,
                            encode_ms=float(prepared_timings.get("encode_ms", estimated_encode_ms)),
                        ),
                    )

                t0 = perf_counter()
                async with AsyncSessionLocal() as db:
                    embedding_repo = EmbeddingRepository(db)
                    batch_top_k = await embedding_repo.find_top_k_batch(
                        [item["prepared"]["embedding"] for item in ok_jobs],
                        k=2,
                    )
                batch_search_ms = (perf_counter() - t0) * 1000.0
                logger.info(
                    "batch_search_done",
                    extra={
                        "batch_size": len(ok_jobs),
                        "batch_search_ms": batch_search_ms,
                    },
                )
                estimated_vector_search_ms = batch_search_ms / max(1, len(ok_jobs))

                for item, top_k in zip(ok_jobs, batch_top_k):
                    item["top_k"] = top_k
                    item["prepared"]["timings"]["vector_search_ms"] = estimated_vector_search_ms
                    item["prepared"]["timings"]["batch_search_wall_ms"] = batch_search_ms

                logger.info(
                    "batch_ready",
                    extra={
                        "batch_size": len(prepared_jobs),
                        "ok_jobs": len(ok_jobs),
                    },
                )
        except Exception as exc:
            # Batch-level сбой (например, общий vector-search по батчу). Причина —
            # в server-лог (traceback, метаданные); клиентам уходит status="error"
            # без деталей (_sanitize, 152-ФЗ).
            logger.exception(
                "verify_batch_failed",
                extra={
                    "batch_size": len(ok_jobs),
                    "error_type": type(exc).__name__,
                },
            )
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
                _complete_terminal_job_inline(
                    job_id=job_id,
                    terminal_state="error",
                    created_at=created_at,
                    job_started_at=job_started_at,
                    dequeued_at=dequeued_at,
                    result_write_ms=result_write_ms,
                    claim_at=float(item.get("worker_claimed_at_ns", now_epoch_ns())) / 1_000_000_000.0,
                    prepared_timings=prepared_timings,
                    vector_search_ms=prepared_timings.get("vector_search_ms"),
                    outcome="error",
                    quality_reason=prepared.get("quality_reason"),
                    quality_details=prepared.get("quality_details", {}),
                    trace_id=_trace_id_from_job_data(item),
                )
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
                job_timings = item.get("job_timings") or StageTimings()
                accepted_at_ns = _payload_timestamp_ns(
                    payload,
                    "accepted_at_ns",
                    int(created_at * 1_000_000_000),
                )
                enqueued_at_ns = _payload_timestamp_ns(payload, "enqueued_at_ns", accepted_at_ns)
                worker_claimed_at_ns = int(item.get("worker_claimed_at_ns", now_epoch_ns()))
                batch_collected_at_ns = int(item.get("batch_collected_at_ns", now_epoch_ns()))
                prepared = item["prepared"]
                prepared_timings = prepared.get("timings", {})
                prepared_timings["batch_verify_loop_wait_ms"] = (
                    perf_counter() - verify_loop_started_at
                ) * 1000.0
                job_timings.set(
                    "queue_wait_ms",
                    max(0.0, (worker_claimed_at_ns - enqueued_at_ns) / 1_000_000.0),
                )
                job_timings.set(
                    "batch_wait_ms",
                    max(0.0, (batch_collected_at_ns - worker_claimed_at_ns) / 1_000_000.0),
                )
                job_timings.set(
                    "worker_semaphore_wait_ms",
                    float(prepared_timings.get("worker_semaphore_wait_ms", 0.0)),
                )

                try:
                    processing_started_at_ns = now_epoch_ns()
                    t_verify = perf_counter()
                    # Клиентский IP из job-payload → contextvar, чтобы create_log
                    # (внутри service) записал request_ip в verification_logs
                    # (audit E2). Reset в finally — не утекает в следующий job.
                    ip_token = set_client_ip(payload.get("request_ip"))
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
                    processing_finished_at_ns = now_epoch_ns()
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
                    normalized_stage_timings = _normalize_job_stage_timings(prepared_timings)
                    prepared_timings.update(normalized_stage_timings)
                    job_timings.set("pipeline_total_ms", normalized_stage_timings["pipeline_total_ms"])
                    _observe_prepared_timings(prepared_timings)
                    result["bbox_source"] = prepared.get("bbox_source")
                    result["bbox_source_detail"] = prepared.get("bbox_source_detail")

                    dequeued_at = item.get("dequeued_at", job_started_at)
                    logger.info(
                        "verify_pipeline_completed",
                        extra=_job_extra(
                            item,
                            preprocess_ms=float(prepared_timings.get("preprocess_ms", 0.0)),
                            quality_gate_pre_ms=float(prepared_timings.get("quality_gate_pre_ms", 0.0)),
                            detect_ms=float(prepared_timings.get("detect_ms", 0.0)),
                            align_crop_ms=float(prepared_timings.get("align_crop_ms", 0.0)),
                            quality_gate_face_ms=float(prepared_timings.get("quality_gate_face_ms", 0.0)),
                            liveness_ms=float(prepared_timings.get("liveness_ms", 0.0)),
                            vector_search_ms=float(prepared_timings.get("vector_search_ms", 0.0)),
                        ),
                    )
                    timestamps = {
                        "accepted_at_ns": accepted_at_ns,
                        "enqueued_at_ns": enqueued_at_ns,
                        "worker_claimed_at_ns": worker_claimed_at_ns,
                        "processing_started_at_ns": processing_started_at_ns,
                        "processing_finished_at_ns": processing_finished_at_ns,
                    }
                    result_written_at_ns = now_epoch_ns()
                    write_metrics = _build_metrics(
                        created_at,
                        job_started_at,
                        time.time(),
                        dequeued_at=dequeued_at,
                    )
                    technical_timestamps = _build_technical_timestamps(
                        queue_popped_at=dequeued_at,
                        worker_started_at_ns=processing_started_at_ns,
                        completed_at_ns=processing_finished_at_ns,
                        result_visible_at_ms=int(result_written_at_ns / 1_000_000),
                    )
                    result_payload = _build_result_payload(result, job_timings, timestamps)
                    result_write_ms = _timed_result_write(
                        VerifyResultStore.set_done,
                        job_id,
                        result_payload,
                        write_metrics,
                        technical_timestamps=technical_timestamps,
                    )
                    job_timings.set("result_write_ms", result_write_ms)
                    job_timings.set(
                        "job_total_server_ms",
                        (result_written_at_ns - accepted_at_ns) / 1_000_000.0,
                    )
                    _complete_terminal_job_inline(
                        job_id=job_id,
                        terminal_state="success",
                        created_at=created_at,
                        job_started_at=job_started_at,
                        dequeued_at=item.get("dequeued_at", job_started_at),
                        result_write_ms=result_write_ms,
                        claim_at=worker_claimed_at_ns / 1_000_000_000.0,
                        job_timings=job_timings,
                        prepared_timings=prepared_timings,
                        vector_search_ms=prepared_timings.get("vector_search_ms"),
                        outcome="ok",
                        quality_reason=prepared.get("quality_reason"),
                        quality_details=prepared.get("quality_details", {}),
                        trace_id=_trace_id_from_job_data(item),
                    )
                except Exception as exc:
                    inc_async_stage_failure("pipeline", exc.__class__.__name__)
                    # Server-side лог причины сбоя verify_from_pipeline_result.
                    # Клиенту уходит только status="error" (детали вырезаются
                    # _sanitize_mapping из result — 152-ФЗ, биометрия/служебное не
                    # утекает в ответ). Здесь — только traceback (метаданные, без
                    # кадров/эмбеддингов), чтобы диагностировать «status=error без
                    # причины» в логах worker'а.
                    logger.exception(
                        "verify_job_failed",
                        extra={
                            "job_id": job_id,
                            "prepared_status": prepared.get("status"),
                            "trace_id": _trace_id_from_job_data(item),
                        },
                    )
                    dequeued_at = item.get("dequeued_at", job_started_at)
                    write_metrics = _build_metrics(created_at, job_started_at, time.time(), dequeued_at=dequeued_at)
                    technical_timestamps = _build_technical_timestamps(
                        queue_popped_at=dequeued_at,
                        worker_started_at_ns=int(job_started_at * 1_000_000_000),
                        completed_at_ns=int(time.time() * 1_000_000_000),
                        result_visible_at_ms=int(time.time() * 1000.0),
                    )
                    result_write_ms = _timed_result_write(
                        VerifyResultStore.set_error,
                        job_id,
                        str(exc),
                        write_metrics,
                        technical_timestamps=technical_timestamps,
                    )
                    _complete_terminal_job_inline(
                        job_id=job_id,
                        terminal_state="error",
                        created_at=created_at,
                        job_started_at=job_started_at,
                        dequeued_at=dequeued_at,
                        result_write_ms=result_write_ms,
                        claim_at=worker_claimed_at_ns / 1_000_000_000.0,
                        job_timings=job_timings,
                        prepared_timings=prepared_timings,
                        vector_search_ms=prepared_timings.get("vector_search_ms"),
                        outcome="error",
                        quality_reason=prepared.get("quality_reason"),
                        quality_details=prepared.get("quality_details", {}),
                        trace_id=_trace_id_from_job_data(item),
                    )
                finally:
                    reset_client_ip(ip_token)

            t_commit = perf_counter()
            await db.commit()
            batch_db_commit_wall_ms = (perf_counter() - t_commit) * 1000.0
            logger.info(
                "batch_db_commit_done",
                extra={
                    "batch_commit_ms": batch_db_commit_wall_ms,
                    "batch_size": len(ok_jobs),
                },
            )

    finally:
        WORKER_ACTIVE_BATCHES.dec()
        # Удаление исходных фото из MinIO после обработки (любой исход: success,
        # quality_reject, spoof, processing_failed, error). Best-effort —
        # MinIO lifecycle-cover на случай падения воркера mid-batch.
        for _job in job_datas:
            try:
                _cleanup_minio_image(
                    _job.get("payload", {}).get("image_url"),
                    _job.get("job_id", ""),
                    stage="process_batch",
                )
            except Exception:
                pass


async def run_worker() -> None:
    """Главная точка входа worker-цикла: setup metrics/threads, warmup, batch-loop.

    Стартует Prometheus HTTP-server (9101), приводит inflight-состояние очереди в
    ноль после рестарта, прогревает ML-pipeline, затем крутит batch-цикл:
    collect_batch → asyncio.Task(process_batch). Семафор batch_slots ограничивает
    параллельные активные батчи (MAX_ACTIVE_BATCH_TASKS).
    """
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    start_http_server(9101)
    _log_service_runtime_snapshot()
    reconciliation = await VerifyJobQueue.reconcile_inflight_state()
    if reconciliation.get("reconciled"):
        logger.info(
            "inflight_reconciled_to_zero",
            extra={
                "queue_len": reconciliation.get("queue_len"),
                "inflight": reconciliation.get("inflight"),
                "claim_count": reconciliation.get("claim_count"),
                "lease_count": reconciliation.get("lease_count"),
            },
        )
    await warmup()

    batch_slots = asyncio.Semaphore(MAX_ACTIVE_BATCH_TASKS)
    active_tasks: set[asyncio.Task[None]] = set()

    while True:
        await batch_slots.acquire()
        try:
            batch = await collect_batch()
        except Exception:
            batch_slots.release()
            raise

        if not batch:
            batch_slots.release()
            await asyncio.sleep(0.001)
            continue

        async def _run(batch_data: list[dict[str, Any]]) -> None:
            try:
                await process_batch(batch_data)
            except Exception:
                extra: dict[str, Any] = {}
                if batch_data:
                    extra = _job_extra(batch_data[0])
                logger.exception("batch_processing_failed", extra=extra)
            finally:
                batch_slots.release()

        task = asyncio.create_task(_run(batch))
        active_tasks.add(task)
        task.add_done_callback(active_tasks.discard)
        await asyncio.sleep(0)


if __name__ == "__main__":
    setup_logging()
    asyncio.run(run_worker())
