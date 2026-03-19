# app/infrastructure/redis_client.py

import redis
from app.core.config import settings


class RedisClient:

    def __init__(self):

        self.client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True
        )

    def get(self, key: str):
        return self.client.get(key)

    def set(self, key: str, value: str, ttl: int = 300):
        self.client.set(key, value, ex=ttl)

    def delete(self, key: str):
        self.client.delete(key)


redis_client = RedisClient()