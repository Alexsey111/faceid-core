import numpy as np
import pytest
from typing import cast

import redis

from app.core.config import settings
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.services.search_service import SearchService


# --- Fake Redis (чтобы не поднимать реальный сервер) ---
class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value


# --- Dummy repo ---
class DummyRepo:
    def __init__(self):
        self.calls = 0

    async def find_top_k(self, embedding, k=2):
        self.calls += 1
        return [{"user_id": 1, "similarity": 0.9}]

    async def get_all_vectors(self):
        return []


def norm(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


@pytest.mark.asyncio
async def test_cache_hit():
    settings.REDIS_ENABLED = True

    repo = DummyRepo()
    service = SearchService(cast(EmbeddingRepository, repo))

    fake = FakeRedis()
    service._redis = cast(redis.Redis, fake)

    query = norm(np.random.rand(512))

    result1 = await service.search_top_k(query, k=1)
    result2 = await service.search_top_k(query, k=1)

    assert result1 == result2
    assert repo.calls == 1


@pytest.mark.asyncio
async def test_cache_key_stability():
    settings.REDIS_ENABLED = True

    repo = DummyRepo()
    service = SearchService(cast(EmbeddingRepository, repo))
    fake = FakeRedis()
    service._redis = cast(redis.Redis, fake)

    query = norm(np.random.rand(512))

    await service.search_top_k(query, k=1)

    assert len(fake.store) == 1


@pytest.mark.asyncio
async def test_cache_different_queries():
    settings.REDIS_ENABLED = True

    repo = DummyRepo()
    service = SearchService(cast(EmbeddingRepository, repo))
    fake = FakeRedis()
    service._redis = cast(redis.Redis, fake)

    q1 = norm(np.random.rand(512))
    q2 = norm(np.random.rand(512))

    await service.search_top_k(q1, k=1)
    await service.search_top_k(q2, k=1)

    assert len(fake.store) == 2


@pytest.mark.asyncio
async def test_cache_disabled():
    settings.REDIS_ENABLED = False

    repo = DummyRepo()
    service = SearchService(cast(EmbeddingRepository, repo))

    query = norm(np.random.rand(512))

    await service.search_top_k(query, k=1)
    await service.search_top_k(query, k=1)

    assert repo.calls == 2


@pytest.mark.asyncio
async def test_redis_failure_does_not_break_search():
    settings.REDIS_ENABLED = True

    repo = DummyRepo()
    service = SearchService(cast(EmbeddingRepository, repo))

    class BrokenRedis:
        def get(self, *args, **kwargs):
            raise Exception("redis down")

        def setex(self, *args, **kwargs):
            raise Exception("redis down")

    service._redis = cast(redis.Redis, BrokenRedis())

    query = norm(np.random.rand(512))

    result = await service.search_top_k(query, k=1)

    assert result
    assert repo.calls == 1
