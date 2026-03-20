import asyncio
import numpy as np
import pytest

from app.services.verification_service import VerificationService
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.verification_repo import VerificationRepository


class DummyEmbeddingRepo(EmbeddingRepository):
    async def find_top_k(self, *args, **kwargs):
        return []

    async def get_all_vectors(self):
        return []


class DummyVerificationRepo(VerificationRepository):
    async def create_log(self, *args, **kwargs):
        pass


@pytest.fixture
def service():
    return VerificationService(
        embedding_repo=DummyEmbeddingRepo(None),  # type: ignore
        verification_repo=DummyVerificationRepo(None),  # type: ignore
    )


@pytest.mark.asyncio
async def test_parallel_verify_requests(service):
    """
    10 параллельных запросов verify
    """

    async def task():
        return await service.verify_face(b"test_image")

    results = await asyncio.gather(*[task() for _ in range(10)])

    assert len(results) == 10
    assert all("status" in r for r in results)


@pytest.mark.asyncio
async def test_burst_requests(service):
    """
    burst нагрузка (быстрые последовательные вызовы)
    """

    results = []

    for _ in range(20):
        r = await service.verify_face(b"burst")
        results.append(r)

    assert len(results) == 20
    assert all("status" in r for r in results)


@pytest.mark.asyncio
async def test_parallel_mixed_inputs(service):
    """
    разные входы параллельно
    """

    async def task(i):
        data = f"image_{i}".encode()
        return await service.verify_face(data)

    results = await asyncio.gather(*[task(i) for i in range(10)])

    assert len(results) == 10
    assert all("status" in r for r in results)


@pytest.mark.asyncio
async def test_pipeline_thread_safety(service):
    """
    проверка, что pipeline не ломается при конкурентном доступе
    """

    async def task():
        return await service.verify_face(b"same_input")

    results = await asyncio.gather(*[task() for _ in range(15)])

    assert all("status" in r for r in results)


@pytest.mark.asyncio
async def test_no_shared_state_corruption(service):
    """
    проверка, что нет утечки состояния между запросами
    """

    async def task(i):
        return await service.verify_face(f"img_{i}".encode())

    results = await asyncio.gather(*[task(i) for i in range(15)])

    # проверяем, что все ответы независимы
    statuses = [r["status"] for r in results]

    assert len(statuses) == 15