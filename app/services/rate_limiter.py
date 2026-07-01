# app\services\rate_limiter.py

from __future__ import annotations

from typing import cast

from fastapi import HTTPException, Request

from app.core.config import settings
from app.infrastructure.redis_client import redis_client
from app.monitoring.metrics import RATE_LIMIT_HITS


def get_inflight_limit(queue_delay_ms: float) -> int:
    if queue_delay_ms < 200:
        return 30
    elif queue_delay_ms < 800:
        return 15
    elif queue_delay_ms < 1500:
        return 9
    else:
        return 5


def get_queue_delay() -> float:
    try:
        return float(redis_client.get("queue_delay_ms") or 0.0)
    except Exception:
        return 0.0


class RateLimiter:
    @staticmethod
    def check(request: Request, key_prefix: str, limit: int, window: int = 60):
        if getattr(settings, "ENV", "").lower() in {"testing", "test"}:
            return

        client = request.client
        if client and client.host:
            client_ip = client.host
        else:
            client_ip = request.headers.get("X-Real-IP")
            if not client_ip:
                forwarded = request.headers.get("X-Forwarded-For")
                client_ip = forwarded.split(",")[0].strip() if forwarded else "unknown"

        key = f"rate:{key_prefix}:{client_ip}"
        count = cast(int, redis_client.incr(key))

        if count == 1:
            redis_client.expire(key, window)

        if count > limit:
            RATE_LIMIT_HITS.inc()
            raise HTTPException(status_code=429, detail="Too many requests")
