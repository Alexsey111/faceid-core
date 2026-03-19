import numpy as np
import pytest

from app.core.config import settings
from app.services.search_service import SearchService


class DummyRepo:
    async def find_top_k(self, *args, **kwargs):
        return []


@pytest.fixture
def reset_faiss():
    original_enabled = settings.FAISS_ENABLED
    SearchService._faiss_index = None
    yield
    settings.FAISS_ENABLED = original_enabled
    SearchService._faiss_index = None


@pytest.mark.asyncio
async def test_faiss_search_returns_indexed_result(reset_faiss):
    settings.FAISS_ENABLED = True
    service = SearchService(DummyRepo())
    vector = np.random.RandomState(0).randn(512).astype(np.float32)

    service.add_embedding(vector, user_id=42)
    results = await service.search_top_k(vector, k=1)

    assert results
    assert results[0]["user_id"] == 42


@pytest.mark.asyncio
async def test_search_falls_back_to_repository(reset_faiss):
    settings.FAISS_ENABLED = False

    fallback = [{"user_id": 7, "similarity": 0.42}]

    class FallbackRepo:
        async def find_top_k(self, *args, **kwargs):
            return fallback

    service = SearchService(FallbackRepo())
    vector = np.zeros(512, dtype=np.float32)

    results = await service.search_top_k(vector, k=1)
    assert results == fallback


def test_add_embedding_increments_index(reset_faiss):
    settings.FAISS_ENABLED = True
    service = SearchService(DummyRepo())

    service.add_embedding(np.zeros(512, dtype=np.float32), user_id=1)
    service.add_embedding(np.ones(512, dtype=np.float32), user_id=2)

    assert SearchService._faiss_index.index.ntotal == 2
    assert SearchService._faiss_index.user_ids == [1, 2]
