from typing import cast

import os
import redis
import time
from fastapi import HTTPException, Request

from app.core.config import settings

RATE_LIMITER_REDIS = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    decode_responses=True,
)


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

        # атомарное увеличение
        count = cast(int, RATE_LIMITER_REDIS.incr(key))

        # если первый запрос — ставим TTL
        if count == 1:
            RATE_LIMITER_REDIS.expire(key, window)

        if count > limit:
            raise HTTPException(
                status_code=429,
                detail="Too many requests"
            )
