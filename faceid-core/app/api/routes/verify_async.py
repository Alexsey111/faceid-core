# app/api/routes/verify_async.py

import base64

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.core.config import settings
from app.infrastructure.redis_client import redis_client
from app.services.verify_job_queue import VerifyJobQueue

router = APIRouter()


class VerifyAsyncRequest(BaseModel):
    image_b64: str
    user_id: str | None = None
    require_liveness: bool = False


@router.post("/verify_async")
async def verify_async(request: VerifyAsyncRequest, http_request: Request):
    print(">>> HIT VERIFY_ASYNC", flush=True)
    print(">>> BEFORE ENQUEUE", flush=True)

    if len(request.image_b64) > 5_000_000:
        raise HTTPException(
            status_code=400,
            detail={"error": "image_too_large"},
        )

    try:
        image_bytes = base64.b64decode(request.image_b64)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_image"},
            )

        h, w = image.shape[:2]
        if min(h, w) < 160:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "image_too_small",
                    "min_size": 160,
                },
            )

        queue_length = int(redis_client.llen(VerifyJobQueue.QUEUE_NAME))
        if queue_length >= VerifyJobQueue.MAX_QUEUE_SIZE:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "overloaded",
                    "reason": "queue_overflow",
                    "retry_after_ms": 500,
                },
            )

        job_id = VerifyJobQueue.enqueue({
            "image_b64": request.image_b64,
            "user_id": request.user_id,
            "require_liveness": request.require_liveness
        })
    except HTTPException:
        raise
    except Exception as e:
        print(f">>> ENQUEUE ERROR: {e}", flush=True)
        raise

    return {
        "job_id": job_id,
        "status": "queued"
    }


@router.post("/ping_async")
async def ping():
    return {"ok": True}
