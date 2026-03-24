# app/infrastructure/redis_client.py

import redis
from typing import cast

from app.core.config import settings


class RedisClient:

    def __init__(self):

        self.client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True
        )

    def get(self, key: str) -> str | None:
        return cast(str | None, self.client.get(key))

    def set(self, key: str, value: str, ttl: int = 300) -> None:
        self.client.set(key, value, ex=ttl)

    def setex(self, key: str, value: str, ttl: int = 300) -> None:
        self.client.setex(key, ttl, value)

    def delete(self, key: str) -> None:
        self.client.delete(key)

    def incr(self, key: str, amount: int = 1) -> int:
        return cast(int, self.client.incr(key, amount))

    def decr(self, key: str, amount: int = 1) -> int:
        return cast(int, self.client.decr(key, amount))

    def expire(self, key: str, ttl: int) -> None:
        self.client.expire(key, ttl)


redis_client = RedisClient()
