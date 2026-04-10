# app/api/routes/verify_async.py

import base64
import binascii
import json
from contextlib import contextmanager
from time import perf_counter

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ValidationError

from app.infrastructure.redis_client import redis_client
from app.monitoring.metrics import (
    ERROR_COUNTER,
    QUEUE_LENGTH,
    QUEUE_PUSH_LATENCY_MS,
    REDIS_COMMAND_LATENCY_MS,
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
)
from app.services.anti_replay_service import AntiReplayService
from app.services.verify_job_queue import VerifyJobQueue

router = APIRouter()


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


@router.post("/verify_async")
async def verify_async(http_request: Request):
    started_total = perf_counter()

    try:
        with observe_ms(VERIFY_ASYNC_BODY_READ_MS):
            body_bytes = await http_request.body()

        VERIFY_ASYNC_REQUEST_SIZE_BYTES.observe(len(body_bytes))

        with observe_ms(VERIFY_ASYNC_JSON_PARSE_MS):
            try:
                payload_dict = json.loads(body_bytes)
            except json.JSONDecodeError:
                VERIFY_ASYNC_REJECTED_TOTAL.labels(reason="invalid_json").inc()
                raise HTTPException(
                    status_code=400,
                    detail={"error": "invalid_json"},
                )

        with observe_ms(VERIFY_ASYNC_MODEL_VALIDATE_MS):
            try:
                payload = VerifyAsyncRequest.model_validate(payload_dict)
            except ValidationError:
                VERIFY_ASYNC_REJECTED_TOTAL.labels(reason="invalid_payload").inc()
                raise HTTPException(
                    status_code=422,
                    detail={"error": "invalid_payload"},
                )

        normalized_b64 = _normalize_image_b64(payload.image_b64)
        VERIFY_ASYNC_IMAGE_B64_CHARS.observe(len(normalized_b64))

        if len(normalized_b64) > 5_000_000:
            VERIFY_ASYNC_REJECTED_TOTAL.labels(reason="image_too_large").inc()
            raise HTTPException(
                status_code=400,
                detail={"error": "image_too_large"},
            )

        with observe_ms(VERIFY_ASYNC_BASE64_DECODE_MS):
            try:
                image_bytes = base64.b64decode(normalized_b64, validate=True)
            except (binascii.Error, ValueError):
                VERIFY_ASYNC_REJECTED_TOTAL.labels(reason="invalid_image_b64").inc()
                raise HTTPException(
                    status_code=400,
                    detail={"error": "invalid_image_b64"},
                )

        image_hash = AntiReplayService.compute_hash(image_bytes)
        VERIFY_ASYNC_IMAGE_BYTES.observe(len(image_bytes))

        with observe_ms(VERIFY_ASYNC_IMAGE_DECODE_MS):
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            VERIFY_ASYNC_REJECTED_TOTAL.labels(reason="invalid_image").inc()
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_image"},
            )

        with observe_ms(VERIFY_ASYNC_PRECHECK_MS):
            h, w = image.shape[:2]
            if min(h, w) < 160:
                VERIFY_ASYNC_REJECTED_TOTAL.labels(reason="image_too_small").inc()
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "image_too_small",
                        "min_size": 160,
                    },
                )

        with observe_ms(REDIS_COMMAND_LATENCY_MS.labels(command="llen")):
            queue_length = int(redis_client.llen(VerifyJobQueue.QUEUE_NAME))

        QUEUE_LENGTH.set(queue_length)

        if queue_length >= VerifyJobQueue.MAX_QUEUE_SIZE:
            VERIFY_ASYNC_REJECTED_TOTAL.labels(reason="queue_overflow").inc()
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "overloaded",
                    "reason": "queue_overflow",
                    "retry_after_ms": 500,
                },
            )

        enqueue_payload = {
            "image_b64": normalized_b64,
            "image_hash": image_hash,
            "user_id": payload.user_id,
            "require_liveness": payload.require_liveness,
        }

        with observe_ms(VERIFY_ASYNC_ENQUEUE_MS), observe_ms(QUEUE_PUSH_LATENCY_MS):
            job_id = VerifyJobQueue.enqueue(enqueue_payload)

        VERIFY_ASYNC_ACCEPTED_TOTAL.inc()

        with observe_ms(VERIFY_ASYNC_RESPONSE_BUILD_MS):
            response = {
                "job_id": job_id,
                "status": "queued",
            }

        return response

    except HTTPException:
        raise

    except Exception as exc:
        ERROR_COUNTER.labels(
            stage="verify_async",
            error_type=type(exc).__name__,
        ).inc()
        raise

    finally:
        VERIFY_ASYNC_ROUTE_MS.observe((perf_counter() - started_total) * 1000)


@router.post("/ping_async")
async def ping():
    return {"ok": True}
