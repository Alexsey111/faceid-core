import numpy as np
import pytest
from typing import cast

from app.core.config import settings
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.services.faiss_index import FaissIndex
from app.services.search_service import SearchService


class DummyRepo:
    def __init__(self, items):
        self.items = items

    async def find_top_k(self, embedding, k=2):
        # эмулируем pgvector
        scored = []
        for item in self.items:
            sim = float(np.dot(embedding, item["embedding"]))
            scored.append({
                "user_id": item["user_id"],
                "similarity": sim
            })
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:k]

    async def get_all_vectors(self):
        return self.items


def norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def sample_data():
    return [
        {"user_id": 1, "embedding": norm(np.random.RandomState(1).randn(512))},
        {"user_id": 2, "embedding": norm(np.random.RandomState(2).randn(512))},
        {"user_id": 3, "embedding": norm(np.random.RandomState(3).randn(512))},
    ]


@pytest.mark.asyncio
async def test_faiss_vs_pgvector_consistency(sample_data):
    settings.FAISS_ENABLED = True
    settings.FAISS_PERSIST_ENABLED = False

    repo = DummyRepo(sample_data)
    service = SearchService(cast(EmbeddingRepository, repo))

    # build FAISS
    index = FaissIndex()
    index.rebuild(sample_data)
    SearchService._faiss_index = index

    query = sample_data[0]["embedding"]

    faiss_result = await service.search_top_k(query, k=1)

    # отключаем FAISS → идём в pgvector
    settings.FAISS_ENABLED = False
    db_result = await service.search_top_k(query, k=1)

    assert faiss_result[0]["user_id"] == db_result[0]["user_id"]


@pytest.mark.asyncio
async def test_pgvector_vs_cpu_fallback_consistency(sample_data):
    settings.FAISS_ENABLED = False

    repo = DummyRepo(sample_data)
    service = SearchService(cast(EmbeddingRepository, repo))

    query = sample_data[1]["embedding"]

    db_result = await service.search_top_k(query, k=1)

    # ломаем find_top_k → forcing CPU fallback
    async def broken_find(*args, **kwargs):
        raise Exception("force fallback")

    repo.find_top_k = broken_find  # type: ignore

    cpu_result = await service.search_top_k(query, k=1)

    assert db_result[0]["user_id"] == cpu_result[0]["user_id"]


@pytest.mark.asyncio
async def test_faiss_vs_cpu_consistency(sample_data):
    settings.FAISS_ENABLED = True
    settings.FAISS_PERSIST_ENABLED = False

    repo = DummyRepo(sample_data)
    service = SearchService(cast(EmbeddingRepository, repo))

    index = FaissIndex()
    index.rebuild(sample_data)
    SearchService._faiss_index = index

    query = sample_data[2]["embedding"]

    faiss_result = await service.search_top_k(query, k=1)

    # отключаем FAISS и pgvector → CPU fallback
    settings.FAISS_ENABLED = False

    async def broken_find(*args, **kwargs):
        raise Exception("force fallback")

    repo.find_top_k = broken_find  # type: ignore

    cpu_result = await service.search_top_k(query, k=1)

    assert faiss_result[0]["user_id"] == cpu_result[0]["user_id"]
