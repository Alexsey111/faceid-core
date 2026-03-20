import numpy as np
import pytest
from typing import cast

from app.core.config import settings
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.services.faiss_index import FaissIndex
from app.services.search_service import SearchService


class DummyRepo:
    async def find_top_k(self, *args, **kwargs):
        return []


@pytest.fixture
def reset_faiss():
    from app.core.config import settings

    settings.FAISS_PERSIST_ENABLED = False  # 🔴 ключевой фикс

    SearchService._faiss_index = FaissIndex()
    yield
    SearchService._faiss_index = None


@pytest.mark.asyncio
async def test_faiss_search_returns_indexed_result(reset_faiss):
    settings.FAISS_ENABLED = True
    service = SearchService(cast(EmbeddingRepository, DummyRepo()))
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
    service = SearchService(cast(EmbeddingRepository, FallbackRepo()))
    vector = np.zeros(512, dtype=np.float32)

    results = await service.search_top_k(vector, k=1)
    assert results == fallback


def test_add_embedding_increments_index(reset_faiss):
    settings.FAISS_ENABLED = True
    service = SearchService(cast(EmbeddingRepository, DummyRepo()))

    service.add_embedding(np.zeros(512, dtype=np.float32), user_id=1)
    service.add_embedding(np.ones(512, dtype=np.float32), user_id=2)

    assert SearchService._faiss_index is not None
    assert SearchService._faiss_index.index.ntotal == 2
    assert SearchService._faiss_index.user_ids == [1, 2]
