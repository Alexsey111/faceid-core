# conftest.py - Конфигурация тестов

import os
import sys
import asyncio
import pytest
import pytest_asyncio
import sqlalchemy as sa

# Переопределяем хосты ДО импорта приложения
os.environ["POSTGRES_HOST"] = "localhost"
os.environ["REDIS_HOST"] = "localhost"
os.environ["MINIO_ENDPOINT"] = "localhost:9000"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'faceid-core'))


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_database():
    from app.db.session import engine

    async with engine.begin() as conn:
        await conn.execute(sa.text(
            "ALTER TABLE verification_logs ADD COLUMN IF NOT EXISTS margin FLOAT DEFAULT 0.0"
        ))
        await conn.execute(sa.text(
            "ALTER TABLE verification_logs ADD COLUMN IF NOT EXISTS liveness_score FLOAT"
        ))
        await conn.execute(sa.text(
            "ALTER TABLE verification_logs ADD COLUMN IF NOT EXISTS is_genuine BOOLEAN"
        ))
        await conn.execute(sa.text(
            "ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS encrypted_embedding BYTEA"
        ))
        await conn.execute(sa.text(
            "ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS embedding BYTEA"
        ))
        await conn.execute(sa.text(
            "ALTER TABLE embeddings ALTER COLUMN embedding DROP NOT NULL"
        ))
    yield


@pytest_asyncio.fixture
async def db_session():
    from app.db.session import AsyncSessionLocal
    async with AsyncSessionLocal() as session:
        yield session
