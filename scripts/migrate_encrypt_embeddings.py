"""Миграция, добавляющая колонку encrypted_embedding и заполняющая её."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


import asyncio

import ast
import numpy as np
from sqlalchemy import select, text

from app.core.config import settings
from app.core.crypto import encrypt_vector
from app.db.session import engine, AsyncSessionLocal
from app.models.embedding import Embedding


async def main():
    print("DATABASE_URL:", settings.DATABASE_URL)
    async with engine.begin() as conn:
        await conn.execute(text(
            "ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS encrypted_embedding BYTEA"
        ))
        await conn.execute(text(
            "ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS embedding BYTEA"
        ))
        await conn.execute(text(
            "ALTER TABLE embeddings ALTER COLUMN embedding DROP NOT NULL"
        ))

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Embedding))
        embeddings = result.scalars().all()
        count = 0
        for embedding in embeddings:
            raw_embedding = embedding.embedding

            if raw_embedding is None:
                continue

            if isinstance(raw_embedding, str):
                raw_embedding = ast.literal_eval(raw_embedding)

            vector = np.array(raw_embedding, dtype=np.float32)
            embedding.encrypted_embedding = encrypt_vector(vector)
            count += 1
        await session.commit()
        print(f"Processed embeddings: {count}")

    await engine.dispose()
    print("Migration completed")


if __name__ == "__main__":
    asyncio.run(main())
