# embedding_repo.py - Репозиторий эмбеддингов

import logging
from typing import Optional, Sequence

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.crypto import decrypt_vector, encrypt_vector
from app.monitoring.metrics import SEARCH_LATENCY
from app.models.embedding import Embedding
from app.models.user import User
from app.monitoring.db_metrics import timed_db_call


class EmbeddingRepository:
    logger = logging.getLogger("EmbeddingRepository")

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_embedding(
        self,
        user_id: int,
        embedding: np.ndarray | list[float]
    ) -> Embedding:
        """
        Сохраняем:
        - raw vector в pgvector для быстрого поиска
        - encrypted_embedding для требований безопасности
        """
        vector = np.asarray(embedding, dtype=np.float32)

        norm = np.linalg.norm(vector)
        if norm == 0.0:
            raise ValueError("Invalid embedding vector")

        vector = vector / norm

        if vector.ndim != 1 or vector.shape[0] != 512:
            raise ValueError("Embedding must be a 512-dim vector")

        encrypted = encrypt_vector(vector)

        record = Embedding(
            user_id=user_id,
            embedding=vector.tolist(),
            encrypted_embedding=encrypted
        )
        self.db.add(record)
        await timed_db_call(self.db.flush(), "embedding_repo.create_embedding.flush")
        await timed_db_call(self.db.refresh(record), "embedding_repo.create_embedding.refresh")
        await timed_db_call(self.db.commit(), "embedding_repo.create_embedding.commit")

        return record

    def _decrypt_record_embedding(self, record: Embedding) -> np.ndarray | None:
        encrypted = record.encrypted_embedding
        if encrypted is None:
            return None

        try:
            return decrypt_vector(encrypted)
        except ValueError as exc:
            self.logger.warning("Skipping embedding id=%s: %s", record.id, exc)
            return None

    async def find_top_k(
        self,
        embedding: np.ndarray,
        k: int = 2
    ) -> list[dict]:
        """
        Быстрый поиск через pgvector по raw embedding.
        Возвращает top-k записей, по одной записи на эмбеддинг.
        """
        vector = np.asarray(embedding, dtype=np.float32)

        norm = np.linalg.norm(vector)
        if norm == 0.0:
            raise ValueError("Invalid embedding vector")

        vector = vector / norm
        if vector.ndim != 1 or vector.shape[0] != 512:
            raise ValueError("Query embedding must be a 512-dim vector")

        embedding_str = "[" + ",".join(str(float(x)) for x in vector) + "]"

        query = text("""
            SELECT
                user_id,
                1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
            FROM embeddings
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :k
        """)

        with SEARCH_LATENCY.time():
            result = await timed_db_call(
                self.db.execute(
                    query,
                    {
                        "embedding": embedding_str,
                        "k": k,
                    },
                ),
                "embedding_repo.find_top_k",
            )
        rows = result.fetchall()

        return [
            {
                "user_id": row.user_id,
                "similarity": float(row.similarity),
            }
            for row in rows
        ]

    async def find_top_k_batch(
        self,
        embeddings: Sequence[np.ndarray],
        k: int = 2,
    ) -> list[list[dict]]:
        """
        Batched pgvector search for multiple query embeddings.
        Returns one top-k list per input embedding, preserving order.
        """
        if not embeddings:
            return []

        normalized_embeddings: list[str] = []
        for embedding in embeddings:
            vector = np.asarray(embedding, dtype=np.float32)

            norm = np.linalg.norm(vector)
            if norm == 0.0:
                raise ValueError("Invalid embedding vector")

            vector = vector / norm
            if vector.ndim != 1 or vector.shape[0] != 512:
                raise ValueError("Query embedding must be a 512-dim vector")

            normalized_embeddings.append("[" + ",".join(str(float(x)) for x in vector) + "]")

        values_sql = ",\n".join(
            f"({idx + 1}, CAST(:embedding_{idx} AS vector))"
            for idx in range(len(normalized_embeddings))
        )

        query = text(f"""
            WITH queries(idx, query_embedding) AS (
                VALUES
                {values_sql}
            )
            SELECT
                q.idx,
                r.user_id,
                r.similarity
            FROM queries q
            JOIN LATERAL (
                SELECT
                    user_id,
                    1 - (embedding <=> q.query_embedding) AS similarity
                FROM embeddings
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> q.query_embedding
                LIMIT :k
            ) r ON true
            ORDER BY q.idx, r.similarity DESC
        """)

        params = {f"embedding_{idx}": value for idx, value in enumerate(normalized_embeddings)}
        params["k"] = k

        with SEARCH_LATENCY.time():
            result = await timed_db_call(
                self.db.execute(query, params),
                "embedding_repo.find_top_k_batch",
            )
        rows = result.fetchall()

        grouped: list[list[dict]] = [[] for _ in embeddings]
        for row in rows:
            grouped[int(row.idx) - 1].append(
                {
                    "user_id": row.user_id,
                    "similarity": float(row.similarity),
                }
            )

        return grouped

    async def find_similar(
        self,
        embedding: np.ndarray,
        k: int = 2,
    ) -> list[dict]:
        return await self.find_top_k(embedding, k=k)

    async def get_by_user_id(self, user_id: int) -> list[Embedding]:
        query = select(Embedding).where(Embedding.user_id == user_id)
        result = await timed_db_call(self.db.execute(query), "embedding_repo.get_by_user_id")
        return list(result.scalars().all())

    async def get_by_user(self, user_id: int) -> list[Embedding]:
        return await self.get_by_user_id(user_id)

    async def delete_by_user_id(self, user_id: int) -> int:
        query = select(Embedding).where(Embedding.user_id == user_id)
        result = await timed_db_call(self.db.execute(query), "embedding_repo.delete_by_user_id")
        embeddings = result.scalars().all()

        count = len(embeddings)
        for emb in embeddings:
            await self.db.delete(emb)
        await timed_db_call(self.db.commit(), "embedding_repo.delete_by_user_id.commit")
        return count

    async def get_all_users_with_embeddings(self) -> list[User]:
        query = select(User).options(selectinload(User.embeddings))
        result = await timed_db_call(self.db.execute(query), "embedding_repo.get_all_users_with_embeddings")
        return list(result.scalars().all())

    async def get_user_vectors(self, user_id: int) -> list[np.ndarray]:
        """
        Для логики centroid в verification лучше использовать raw vector из БД.
        Если raw отсутствует — fallback на decrypt.
        """
        query = select(Embedding).where(Embedding.user_id == user_id)
        result = await timed_db_call(self.db.execute(query), "embedding_repo.get_user_vectors")
        rows = list(result.scalars().all())

        vectors: list[np.ndarray] = []
        for row in rows:
            if row.embedding is not None:
                vectors.append(np.asarray(row.embedding, dtype=np.float32))
                continue

            decrypted = self._decrypt_record_embedding(row)
            if decrypted is not None:
                vectors.append(decrypted)

        return vectors

    async def get_all_vectors(self) -> list[dict]:
        """
        Аварийный fallback для search service.
        Предпочитаем raw embedding, иначе decrypt.
        """
        query = select(
            Embedding.user_id,
            Embedding.embedding,
            Embedding.encrypted_embedding,
            Embedding.id,
        )
        result = await timed_db_call(self.db.execute(query), "embedding_repo.get_all_vectors")
        rows = result.fetchall()

        items: list[dict] = []
        for row in rows:
            if row.embedding is not None:
                items.append({
                    "user_id": row.user_id,
                    "embedding": np.asarray(row.embedding, dtype=np.float32),
                })
                continue

            if row.encrypted_embedding is None:
                continue

            try:
                vector = decrypt_vector(row.encrypted_embedding)
            except ValueError as exc:
                self.logger.warning("Skipping row id=%s: %s", row.id, exc)
                continue

            items.append({
                "user_id": row.user_id,
                "embedding": vector,
            })

        return items
