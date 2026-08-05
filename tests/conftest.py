# conftest.py - test configuration

import asyncio
import os
import subprocess
import sys

import pytest
import pytest_asyncio
import redis as redis_lib
from sqlalchemy.ext.asyncio import create_async_engine

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

os.environ["POSTGRES_HOST"] = "localhost"
os.environ["REDIS_HOST"] = "localhost"
os.environ["MINIO_ENDPOINT"] = "localhost:9000"
os.environ["MODELS_DIR"] = os.path.join(ROOT_DIR, "models")
os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:postgres@localhost:5433/postgres"
# Аутентификация отключена в тестах (тестируется отдельно в test_auth.py,
# который поднимает AUTH_ENABLED=True в рамках своего fixture).
os.environ["AUTH_ENABLED"] = "false"

from app.core.config import settings

settings.FAISS_PERSIST_ENABLED = False
settings.DB_POOL_SIZE = 1
settings.DB_MAX_OVERFLOW = 0
settings.MODELS_DIR = os.path.join(ROOT_DIR, "models")

TEST_DB_URL = os.environ["DATABASE_URL"]
test_engine = create_async_engine(
    TEST_DB_URL,
    pool_size=1,
    max_overflow=0,
)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: pure unit test — no DB/Redis/infra required")


def _all_unit(session) -> bool:
    """True, если в текущем прогоне собраны только unit-тесты (маркер 'unit')."""
    items = getattr(session, "items", None)
    if not items:
        return False
    return all(
        _is_unit_item(item)
        for item in items
    )


def _is_unit_item(item) -> bool:
    return (
        item.get_closest_marker("unit") is not None
        or "/tests/unit/" in str(item.path).replace("\\", "/")
    )


@pytest.fixture(scope="session", autouse=True)
def run_migrations(request):
    # Прогон только unit-тестов не требует схемы БД — пропускаем alembic,
    # чтобы юнит-тесты работали без поднятого Postgres.
    if _all_unit(request.session):
        return
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True, cwd=ROOT_DIR)


@pytest.fixture(autouse=True)
def reset_redis(request):
    if _is_unit_item(request.node):
        yield
        return
    client = redis_lib.Redis(host="localhost", port=6379, db=0)
    client.flushdb()
    yield


@pytest.fixture(autouse=True)
def disable_quality_gate_nonunit(request):
    """E2E/integration-тесты проверяют бизнес-логику (match/no_match), а не
    quality-gate. Тестовые фото (tests/data, tests/images) — мелкие кропы, на
    которых V2 quality-gate режет по face_too_small → 400 quality_reject.
    Отключаем gate для non-unit (unit-тесты качества живут в
    test_upload_validation/test_pipeline_* и gate-логику тестируют сами).
    Выставляем ДО первого pipeline.process() — pipeline читает
    settings.QUALITY_GATE_MODE в _init (lazy)."""
    if _is_unit_item(request.node):
        yield
        return
    prev = settings.QUALITY_GATE_MODE
    settings.QUALITY_GATE_MODE = "off"
    try:
        yield
    finally:
        settings.QUALITY_GATE_MODE = prev


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_database(run_migrations):
    yield


@pytest_asyncio.fixture
async def db_session():
    """DB-сессия для integration-тестов. Не используется в unit-наборе (моки)."""
    from app.db.session import AsyncSessionLocal
    async with AsyncSessionLocal() as session:
        yield session
