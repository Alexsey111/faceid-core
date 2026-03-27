# app/api/routes/verify_async.py

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.core.config import settings
from app.services.rate_limiter import RateLimiter
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

    rate_limit = max(20, int(settings.ASYNC_THROUGHPUT_PER_SEC * 4))

    try:
        RateLimiter.check(http_request, "verify_async_enqueue", limit=rate_limit, window=1)
    except HTTPException:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "too_many_requests",
                "reason": "verify_async_enqueue_rate_limited",
                "retry_after_ms": 500,
            },
        )

    try:
        job_id = VerifyJobQueue.enqueue({
            "image_b64": request.image_b64,
            "user_id": request.user_id,
            "require_liveness": request.require_liveness
        })
    except Exception as e:
        print(f">>> ENQUEUE ERROR: {e}", flush=True)
        if str(e) in {"queue_overflow", "overloaded", "inflight_limit_exceeded", "sla_overflow"}:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "overloaded",
                    "reason": str(e),
                    "retry_after_ms": 500,
                },
            )

        raise

    return {
        "job_id": job_id,
        "status": "queued"
    }


@router.post("/ping_async")
async def ping():
    return {"ok": True}
