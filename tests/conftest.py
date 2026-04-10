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
FACEID_DIR = os.path.join(ROOT_DIR, "faceid-core")
if FACEID_DIR not in sys.path:
    sys.path.insert(0, FACEID_DIR)

os.environ["POSTGRES_HOST"] = "localhost"
os.environ["REDIS_HOST"] = "localhost"
os.environ["CELERY_BROKER_URL"] = "redis://localhost:6379/0"
os.environ["CELERY_RESULT_BACKEND"] = "redis://localhost:6379/0"
os.environ["MINIO_ENDPOINT"] = "localhost:9000"
os.environ["MODELS_DIR"] = os.path.join(ROOT_DIR, "models")
os.environ["USE_PIPELINE_V2"] = "false"
os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:postgres@localhost:5433/postgres"

from app.core.config import settings

settings.FAISS_PERSIST_ENABLED = False
settings.DB_POOL_SIZE = 1
settings.DB_MAX_OVERFLOW = 0
settings.CELERY_BROKER_URL = "redis://localhost:6379/0"
settings.CELERY_RESULT_BACKEND = "redis://localhost:6379/0"
settings.MODELS_DIR = os.path.join(ROOT_DIR, "models")
settings.USE_PIPELINE_V2 = False

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


@pytest.fixture(scope="session", autouse=True)
def run_migrations():
    subprocess.run(["alembic", "upgrade", "head"], check=True, cwd=ROOT_DIR)


@pytest.fixture(autouse=True)
def reset_redis():
    client = redis_lib.Redis(host="localhost", port=6379, db=0)
    client.flushdb()
    yield


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_database(run_migrations):
    yield


@pytest_asyncio.fixture
async def db_session():
    from app.db.session import AsyncSessionLocal
    async with AsyncSessionLocal() as session:
        yield session
