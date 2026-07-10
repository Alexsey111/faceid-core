# app/api/routes/verify_async.py

import asyncio
import base64
import binascii
import json
import logging
import os
import uuid
from contextlib import contextmanager
from enum import StrEnum
from time import perf_counter

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ValidationError

from app.api._helpers import get_request_id
from app.core.config import settings
from app.core.timing import StageTimings, elapsed_ms, now_epoch_ns, now_perf_ns
from app.infrastructure.minio_client import MinioClient
from app.monitoring.metrics import (
    ERROR_COUNTER,
    inc_async_stage_failure,
    VERIFY_ADMISSION_ACCEPTED_TOTAL,
    VERIFY_ADMISSION_ATTEMPTS_TOTAL,
    VERIFY_ADMISSION_ERRORS_TOTAL,
    VERIFY_ADMISSION_ESTIMATED_DELAY_MS,
    VERIFY_ADMISSION_INFLIGHT_SNAPSHOT,
    VERIFY_ADMISSION_QUEUE_LEN_SNAPSHOT,
    VERIFY_ADMISSION_REJECTED_TOTAL,
    VERIFY_ADMISSION_STAGE_MS,
    VERIFY_ASYNC_ACCEPTED_TOTAL,
    VERIFY_ASYNC_BASE64_DECODE_MS,
    VERIFY_ASYNC_BODY_READ_MS,
    VERIFY_ASYNC_ENQUEUE_MS,
    VERIFY_ASYNC_IMAGE_B64_CHARS,
    VERIFY_ASYNC_IMAGE_BYTES,
    VERIFY_ASYNC_IMAGE_DECODE_MS,
    VERIFY_ASYNC_JSON_PARSE_MS,
    VERIFY_ASYNC_MODEL_VALIDATE_MS,
    VERIFY_ASYNC_PRECHECK_MS,
    VERIFY_ASYNC_REJECTED_TOTAL,
    VERIFY_ASYNC_REQUEST_SIZE_BYTES,
    VERIFY_ASYNC_RESPONSE_BUILD_MS,
    VERIFY_ASYNC_ROUTE_MS,
    observe_async_stage,
)
from app.services.verify_job_queue import VerifyJobQueue

router = APIRouter()
logger = logging.getLogger(__name__)
VERIFY_ROUTE = "verify_async"
ADMISSION_DEBUG_HEADERS = os.getenv("ADMISSION_DEBUG_HEADERS", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class AdmissionRejectReason(StrEnum):
    INVALID_REQUEST = "invalid_request"
    INVALID_BASE64 = "invalid_base64"
    IMAGE_DECODE_FAILED = "image_decode_failed"
    IMAGE_TOO_LARGE = "image_too_large"
    IMAGE_TOO_SMALL = "image_too_small"
    PRECHECK_FAILED = "precheck_failed"
    INFLIGHT_LIMIT = "inflight_limit"
    QUEUE_OVERFLOW = "queue_overflow"
    ESTIMATED_DELAY = "estimated_delay"
    SLA_OVERFLOW = "sla_overflow"
    ADMISSION_GUARD = "admission_guard"
    ENQUEUE_ERROR = "enqueue_error"


class AdmissionRejected(HTTPException):
    def __init__(
        self,
        *,
        reason: AdmissionRejectReason,
        stage: str,
        status_code: int,
        detail: dict[str, object],
        request_id: str,
        job_id: str | None = None,
        queue_len: int | None = None,
        inflight: int | None = None,
        estimated_delay_ms: float | None = None,
        max_queue_size: int | None = None,
        inflight_limit: int | None = None,
        throughput_per_sec: float | None = None,
    ) -> None:
        headers = None
        if ADMISSION_DEBUG_HEADERS:
            headers = {
                "X-Admission-Reason": reason.value,
                "X-Admission-Queue-Len": _fmt_header_int(queue_len),
                "X-Admission-Inflight": _fmt_header_int(inflight),
                "X-Admission-Estimated-Delay-Ms": _fmt_header_float(estimated_delay_ms),
            }
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.reason = reason
        self.stage = stage
        self.request_id = request_id
        self.job_id = job_id
        self.queue_len = queue_len
        self.inflight = inflight
        self.estimated_delay_ms = estimated_delay_ms
        self.max_queue_size = max_queue_size
        self.inflight_limit = inflight_limit
        self.throughput_per_sec = throughput_per_sec


def _fmt_header_int(value: int | None) -> str:
    return "" if value is None else str(int(value))


def _fmt_header_float(value: float | None) -> str:
    return "" if value is None else f"{float(value):.3f}"


class VerifyAsyncRequest(BaseModel):
    image_b64: str
    user_id: str | None = None
    require_liveness: bool = False


@contextmanager
def observe_ms(metric):
    started = perf_counter()
    try:
        yield
    finally:
        metric.observe((perf_counter() - started) * 1000)


def _normalize_image_b64(image_b64: str) -> str:
    if image_b64.startswith("data:"):
        _, _, tail = image_b64.partition(",")
        return tail or image_b64
    return image_b64


def _record_async_timings(timings: StageTimings) -> None:
    for stage, value_ms in timings.values.items():
        stage_name = stage[:-3] if stage.endswith("_ms") else stage
        observe_async_stage(stage_name, value_ms)


@contextmanager
def _observe_admission_stage(stage: str):
    started = perf_counter()
    outcome = "accepted"
    try:
        yield
    except AdmissionRejected:
        outcome = "rejected"
        raise
    except Exception:
        outcome = "error"
        raise
    finally:
        VERIFY_ADMISSION_STAGE_MS.labels(
            route=VERIFY_ROUTE,
            stage=stage,
            outcome=outcome,
        ).observe((perf_counter() - started) * 1000.0)


def _log_admission_rejected(exc: AdmissionRejected) -> None:
    logger.warning(
        "admission_rejected",
        extra={
            "route": VERIFY_ROUTE,
            "reason": exc.reason.value,
            "job_id": exc.job_id,
            "request_id": exc.request_id,
            "trace_id": exc.request_id,
            "queue_len": exc.queue_len,
            "inflight": exc.inflight,
            "estimated_delay_ms": exc.estimated_delay_ms,
            "max_queue_size": exc.max_queue_size,
            "inflight_limit": exc.inflight_limit,
            "throughput_per_sec": exc.throughput_per_sec,
            "stage": exc.stage,
        },
    )


def _raise_admission_rejected(
    *,
    stage: str,
    reason: AdmissionRejectReason,
    status_code: int,
    detail: dict[str, object],
    request_id: str,
    job_id: str | None = None,
    queue_len: int | None = None,
    inflight: int | None = None,
    estimated_delay_ms: float | None = None,
    max_queue_size: int | None = None,
    inflight_limit: int | None = None,
    throughput_per_sec: float | None = None,
) -> None:
    raise AdmissionRejected(
        reason=reason,
        stage=stage,
        status_code=status_code,
        detail=detail,
        request_id=request_id,
        job_id=job_id,
        queue_len=queue_len,
        inflight=inflight,
        estimated_delay_ms=estimated_delay_ms,
        max_queue_size=max_queue_size,
        inflight_limit=inflight_limit,
        throughput_per_sec=throughput_per_sec,
    )


def _map_queue_overloaded_reason(reason: str) -> AdmissionRejectReason:
    mapping = {
        "inflight_overflow": AdmissionRejectReason.INFLIGHT_LIMIT,
        "inflight_limit": AdmissionRejectReason.INFLIGHT_LIMIT,
        "queue_overflow": AdmissionRejectReason.QUEUE_OVERFLOW,
        "estimated_delay": AdmissionRejectReason.ESTIMATED_DELAY,
        "sla_overflow": AdmissionRejectReason.SLA_OVERFLOW,
    }
    return mapping.get(reason, AdmissionRejectReason.ADMISSION_GUARD)


@router.post("/verify_async")
async def verify_async(http_request: Request):
    route_start_ns = now_perf_ns()
    timings = StageTimings()
    accepted_at_ns = now_epoch_ns()
    started_total = perf_counter()
    route = VERIFY_ROUTE
    request_id = get_request_id(http_request)
    job_id: str | None = None
    current_stage = "request_parse"

    VERIFY_ADMISSION_ATTEMPTS_TOTAL.labels(route=route).inc()

    try:
        with _observe_admission_stage("request_parse"):
            with observe_ms(VERIFY_ASYNC_BODY_READ_MS):
                body_bytes = await http_request.body()

            VERIFY_ASYNC_REQUEST_SIZE_BYTES.observe(len(body_bytes))

            with observe_ms(VERIFY_ASYNC_JSON_PARSE_MS):
                try:
                    payload_dict = json.loads(body_bytes)
                except json.JSONDecodeError:
                    _raise_admission_rejected(
                        stage="request_parse",
                        reason=AdmissionRejectReason.INVALID_REQUEST,
                        status_code=400,
                        detail={"error": AdmissionRejectReason.INVALID_REQUEST.value},
                        request_id=request_id,
                    )

            with observe_ms(VERIFY_ASYNC_MODEL_VALIDATE_MS):
                try:
                    payload = VerifyAsyncRequest.model_validate(payload_dict)
                except ValidationError:
                    _raise_admission_rejected(
                        stage="request_parse",
                        reason=AdmissionRejectReason.INVALID_REQUEST,
                        status_code=422,
                        detail={"error": AdmissionRejectReason.INVALID_REQUEST.value},
                        request_id=request_id,
                    )

        # Active-challenge gate допуска: /verify_async не несёт liveness_token
        # (схема VerifyAsyncRequest без liveness_mode/token). При
        # LIVENESS_ACTIVE_REQUIRED=true и require_liveness=true — 403 с
        # направлением на /verify_base64 (active proof работает там). До
        # base64-decode и MinIO-upload, чтобы не делать работу ради отказа.
        if payload.require_liveness and settings.LIVENESS_ACTIVE_REQUIRED:
            raise HTTPException(
                status_code=403,
                detail=(
                    "active_liveness_required: this path does not carry liveness_token; "
                    "use /api/v1/verify_base64 with liveness_mode=active + liveness_token"
                ),
            )

        current_stage = "base64_decode"
        with _observe_admission_stage("base64_decode"):
            normalized_b64 = _normalize_image_b64(payload.image_b64)
            VERIFY_ASYNC_IMAGE_B64_CHARS.observe(len(normalized_b64))

            if len(normalized_b64) > 5_000_000:
                _raise_admission_rejected(
                    stage="base64_decode",
                    reason=AdmissionRejectReason.IMAGE_TOO_LARGE,
                    status_code=400,
                    detail={
                        "error": AdmissionRejectReason.IMAGE_TOO_LARGE.value,
                        "max_length": 5_000_000,
                    },
                    request_id=request_id,
                )

            decode_start_ns = now_perf_ns()
            with observe_ms(VERIFY_ASYNC_BASE64_DECODE_MS):
                try:
                    image_bytes = base64.b64decode(normalized_b64, validate=True)
                except (binascii.Error, ValueError):
                    _raise_admission_rejected(
                        stage="base64_decode",
                        reason=AdmissionRejectReason.INVALID_BASE64,
                        status_code=400,
                        detail={"error": AdmissionRejectReason.INVALID_BASE64.value},
                        request_id=request_id,
                    )
            timings.finish("base64_decode_ms", decode_start_ns)

            VERIFY_ASYNC_IMAGE_BYTES.observe(len(image_bytes))

        current_stage = "image_decode"
        with _observe_admission_stage("image_decode"):
            precheck_start_ns = now_perf_ns()
            with observe_ms(VERIFY_ASYNC_IMAGE_DECODE_MS):
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                _raise_admission_rejected(
                    stage="image_decode",
                    reason=AdmissionRejectReason.IMAGE_DECODE_FAILED,
                    status_code=400,
                    detail={"error": AdmissionRejectReason.IMAGE_DECODE_FAILED.value},
                    request_id=request_id,
                )

        current_stage = "precheck"
        with _observe_admission_stage("precheck"):
            try:
                with observe_ms(VERIFY_ASYNC_PRECHECK_MS):
                    h, w = image.shape[:2]
                    if min(h, w) < 160:
                        _raise_admission_rejected(
                            stage="precheck",
                            reason=AdmissionRejectReason.IMAGE_TOO_SMALL,
                            status_code=400,
                            detail={
                                "error": AdmissionRejectReason.IMAGE_TOO_SMALL.value,
                                "min_size": 160,
                            },
                            request_id=request_id,
                        )
            except AdmissionRejected:
                raise
            except Exception:
                _raise_admission_rejected(
                    stage="precheck",
                    reason=AdmissionRejectReason.PRECHECK_FAILED,
                    status_code=400,
                    detail={"error": AdmissionRejectReason.PRECHECK_FAILED.value},
                    request_id=request_id,
                )

            timings.finish("precheck_ms", precheck_start_ns)
            timings.set("admission_ms", elapsed_ms(route_start_ns))

        current_stage = "admission_guard"
        with _observe_admission_stage("admission_guard"):
            try:
                decision = await VerifyJobQueue.evaluate_admission()
            except Exception:
                _raise_admission_rejected(
                    stage="admission_guard",
                    reason=AdmissionRejectReason.ADMISSION_GUARD,
                    status_code=429,
                    detail={"error": AdmissionRejectReason.ADMISSION_GUARD.value},
                    request_id=request_id,
                    max_queue_size=VerifyJobQueue.max_queue_size(),
                    inflight_limit=VerifyJobQueue.inflight_limit(),
                    throughput_per_sec=VerifyJobQueue.async_throughput_per_sec(),
                )

            decision_reason = decision.reason or "accepted"
            VERIFY_ADMISSION_INFLIGHT_SNAPSHOT.labels(
                route=route,
                decision=decision_reason,
            ).observe(float(decision.inflight or 0))
            VERIFY_ADMISSION_QUEUE_LEN_SNAPSHOT.labels(
                route=route,
                decision=decision_reason,
            ).observe(float(decision.queue_len or 0))
            VERIFY_ADMISSION_ESTIMATED_DELAY_MS.labels(
                route=route,
                decision=decision_reason,
            ).observe(float(decision.estimated_delay_ms or 0.0))

            if not decision.accepted:
                reason = _map_queue_overloaded_reason(decision_reason)
                detail: dict[str, object] = {
                    "error": "overloaded",
                    "reason": reason.value,
                }
                if reason == AdmissionRejectReason.INFLIGHT_LIMIT:
                    detail.update(
                        {
                            "inflight": decision.inflight,
                            "inflight_limit": VerifyJobQueue.inflight_limit(),
                            "queue_length": decision.queue_len,
                        }
                    )
                elif reason == AdmissionRejectReason.QUEUE_OVERFLOW:
                    detail.update(
                        {
                            "queue_length": decision.queue_len,
                            "queue_limit": VerifyJobQueue.max_queue_size(),
                            "inflight": decision.inflight,
                        }
                    )
                else:
                    detail.update(
                        {
                            "queue_length": decision.queue_len,
                            "inflight": decision.inflight,
                            "estimated_delay_ms": round(decision.estimated_delay_ms or 0.0, 2),
                            "max_queue_delay_ms": VerifyJobQueue.backpressure_max_queue_delay_ms(),
                        }
                    )
                _raise_admission_rejected(
                    stage="admission_guard",
                    reason=reason,
                    status_code=429,
                    detail=detail,
                    request_id=request_id,
                    queue_len=decision.queue_len,
                    inflight=decision.inflight,
                    estimated_delay_ms=decision.estimated_delay_ms,
                    max_queue_size=VerifyJobQueue.max_queue_size(),
                    inflight_limit=VerifyJobQueue.inflight_limit(),
                    throughput_per_sec=VerifyJobQueue.async_throughput_per_sec(),
                )

        current_stage = "enqueue"
        enqueue_start_ns = now_perf_ns()
        enqueued_at_ns = now_epoch_ns()

        # MinIO upload (как sync-путь _enqueue_verify_job): plaintext base64 НЕ
        # кладём в Redis-очередь (152-ФЗ — исходные фото не хранятся в очереди).
        # Воркер скачивает по image_url и удаляет объект после обработки; lifecycle
        # MinIO-cover на случай падения воркера.
        object_name = f"verify_async/{uuid.uuid4().hex}/image.jpg"
        minio_client = MinioClient()
        uploaded = False
        try:
            await asyncio.to_thread(
                minio_client.upload_image, object_name, image_bytes, "image/jpeg"
            )
            uploaded = True
        except Exception as exc:
            logger.exception(
                "minio_upload_failed",
                extra={"route": route, "request_id": request_id},
            )
            _raise_admission_rejected(
                stage="enqueue",
                reason=AdmissionRejectReason.ENQUEUE_ERROR,
                status_code=503,
                detail={
                    "error": AdmissionRejectReason.ENQUEUE_ERROR.value,
                    "error_type": type(exc).__name__,
                },
                request_id=request_id,
                max_queue_size=VerifyJobQueue.max_queue_size(),
                inflight_limit=VerifyJobQueue.inflight_limit(),
                throughput_per_sec=VerifyJobQueue.async_throughput_per_sec(),
            )

        enqueue_payload = {
            "image_url": object_name,
            "user_id": payload.user_id,
            "require_liveness": payload.require_liveness,
            "accepted_at_ns": accepted_at_ns,
            "enqueued_at_ns": enqueued_at_ns,
            "trace_id": request_id,
        }

        with _observe_admission_stage("enqueue"):
            enqueue_ok = False
            try:
                with observe_ms(VERIFY_ASYNC_ENQUEUE_MS):
                    job_id = await VerifyJobQueue.enqueue_job(enqueue_payload, admission=decision)
                enqueue_ok = True
            except AdmissionRejected:
                raise
            except Exception as exc:
                logger.exception(
                    "enqueue_failed",
                    extra={
                        "route": route,
                        "reason": AdmissionRejectReason.ENQUEUE_ERROR.value,
                        "request_id": request_id,
                        "queue_len": decision.queue_len,
                        "inflight": decision.inflight,
                    },
                )
                _raise_admission_rejected(
                    stage="enqueue",
                    reason=AdmissionRejectReason.ENQUEUE_ERROR,
                    status_code=503,
                    detail={
                        "error": AdmissionRejectReason.ENQUEUE_ERROR.value,
                        "error_type": type(exc).__name__,
                    },
                    request_id=request_id,
                    queue_len=decision.queue_len,
                    inflight=decision.inflight,
                    estimated_delay_ms=decision.estimated_delay_ms,
                    max_queue_size=VerifyJobQueue.max_queue_size(),
                    inflight_limit=VerifyJobQueue.inflight_limit(),
                    throughput_per_sec=VerifyJobQueue.async_throughput_per_sec(),
                )
            finally:
                # enqueue упал (overflow/ошибка) — удаляем загруженный объект,
                # иначе утечка до срабатывания MinIO lifecycle.
                if not enqueue_ok and uploaded:
                    try:
                        await asyncio.to_thread(minio_client.delete_image, object_name)
                    except Exception:
                        logger.warning(
                            "minio_delete_failed_on_enqueue_error image_url=%s",
                            object_name,
                            exc_info=True,
                        )

            timings.finish("enqueue_ms", enqueue_start_ns)

        current_stage = "response_build"
        with _observe_admission_stage("response_build"):
            with observe_ms(VERIFY_ASYNC_RESPONSE_BUILD_MS):
                response = {
                    "job_id": job_id,
                    "status": "queued",
                }

        VERIFY_ADMISSION_ACCEPTED_TOTAL.labels(route=route).inc()
        VERIFY_ASYNC_ACCEPTED_TOTAL.inc()

        return response

    except AdmissionRejected as exc:
        VERIFY_ADMISSION_REJECTED_TOTAL.labels(route=route, reason=exc.reason.value).inc()
        VERIFY_ASYNC_REJECTED_TOTAL.labels(reason=exc.reason.value).inc()
        inc_async_stage_failure("admission", exc.reason.value)
        _log_admission_rejected(exc)
        raise

    except HTTPException:
        raise

    except Exception as exc:
        VERIFY_ADMISSION_ERRORS_TOTAL.labels(route=route, stage=current_stage).inc()
        ERROR_COUNTER.labels(
            stage="verify_async",
            error_type=type(exc).__name__,
        ).inc()
        raise

    finally:
        timings.set("route_total_ms", elapsed_ms(route_start_ns))
        _record_async_timings(timings)
        VERIFY_ASYNC_ROUTE_MS.observe((perf_counter() - started_total) * 1000)
