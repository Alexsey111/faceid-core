from typing import cast

import time
from fastapi import HTTPException, Request

from app.infrastructure.redis_client import redis_client


class RateLimiter:

    @staticmethod
    def check(request: Request, key_prefix: str, limit: int, window: int = 60):

        client = request.client
        if client and client.host:
            client_ip = client.host
        else:
            client_ip = request.headers.get("X-Real-IP")
            if not client_ip:
                forwarded = request.headers.get("X-Forwarded-For")
                client_ip = forwarded.split(",")[0].strip() if forwarded else "unknown"
        key = f"rate:{key_prefix}:{client_ip}"

        print("RATE LIMIT CALLED", key)

        # атомарное увеличение
        count = cast(int, redis_client.client.incr(key))

        # если первый запрос — ставим TTL
        if count == 1:
            redis_client.client.expire(key, window)

        print(f"[RATE LIMIT] key={key} count={count}")

        if count > limit:
            raise HTTPException(
                status_code=429,
                detail="Too many requests"
            )
