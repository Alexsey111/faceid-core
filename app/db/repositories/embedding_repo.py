# embedding_repo.py - Репозиторий эмбеддингов

import logging
from typing import Sequence

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.crypto import decrypt_vector, encrypt_vector, hash_vector
from app.monitoring.metrics import SEARCH_DECRYPT_ALL_FALLBACK_N, SEARCH_LATENCY
from app.models.embedding import Embedding
from app.models.user import User
from app.monitoring.db_metrics import timed_db_call


# Порог для предупреждения о деградации SQL-fallback (decrypt-all).
# Основной путь поиска — FAISS; fallback срабатывает редко. При росте числа
# пользователей N > порога decrypt-all становится заметным — логируем.
_DECRYPT_ALL_WARN_THRESHOLD = 5000


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
        Сохраняем ТОЛЬКО зашифрованный эмбеддинг (AES-256-GCM).
        Plaintext-копия больше не хранится (ТЗ: шаблон — в зашифрованном виде).
        Вектор нормируется L2 до шифрования, поэтому decrypt даёт готовый к
        cosine-сравнению unit-вектор.
        """
        vector = np.asarray(embedding, dtype=np.float32)

        norm = np.linalg.norm(vector)
        if norm == 0.0:
            raise ValueError("Invalid embedding vector")

        vector = vector / norm

        if vector.ndim != 1 or vector.shape[0] != 512:
            raise ValueError("Embedding must be a 512-dim vector")

        encrypted = encrypt_vector(vector)
        # Content-hash от plaintext (ТЗ-схема: encrypted_hash TEXT NOT NULL).
        # Даёт idempotency/lookup без decrypt-all.
        emb_hash = hash_vector(vector)

        record = Embedding(
            user_id=user_id,
            encrypted_embedding=encrypted,
            encrypted_hash=emb_hash,
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

    async def _aload_all_vectors_matrix(
        self, label: str
    ) -> tuple[np.ndarray | None, list[int]]:
        """
        Async-версия: SELECT user_id, encrypted_embedding; decrypt-all в матрицу.
        Возвращает (matrix (N,512) или None, user_ids). Метрика+warning при большом N.
        """
        query = select(Embedding.user_id, Embedding.encrypted_embedding)
        with SEARCH_LATENCY.time():
            result = await timed_db_call(self.db.execute(query), label)
        rows = result.fetchall()

        if not rows:
            return None, []

        n = len(rows)
        if n > _DECRYPT_ALL_WARN_THRESHOLD:
            self.logger.warning(
                "%s: decrypt-all fallback N=%s (FAISS — основной путь поиска)", label, n
            )
        try:
            SEARCH_DECRYPT_ALL_FALLBACK_N.set(n)
        except Exception:
            pass

        matrix = np.empty((n, 512), dtype=np.float32)
        user_ids: list[int] = []
        valid = 0
        for row in rows:
            if row.encrypted_embedding is None:
                continue
            try:
                v = decrypt_vector(row.encrypted_embedding)
            except ValueError as exc:
                self.logger.warning("Skipping embedding decrypt: %s", exc)
                continue
            matrix[valid] = v
            user_ids.append(int(row.user_id))
            valid += 1

        if valid == 0:
            return None, []

        matrix = matrix[:valid]
        # Защитная L2-нормализация (векторы уже нормализованы при записи).
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms
        return matrix, user_ids

    @staticmethod
    def _topk_from_scores(scores: np.ndarray, user_ids: list[int], k: int) -> list[dict]:
        n = len(user_ids)
        if n == 0:
            return []
        kk = max(1, min(k, n))
        if kk == 1:
            idx = np.array([int(np.argmax(scores))])
        else:
            part = np.argpartition(-scores, kk - 1)[:kk]
            idx = part[np.argsort(-scores[part])]
        return [
            {"user_id": int(user_ids[i]), "similarity": float(scores[i])}
            for i in idx
        ]

    async def find_top_k(
        self,
        embedding: np.ndarray,
        k: int = 2
    ) -> list[dict]:
        """
        SQL-fallback поиска: decrypt-all + numpy cosine.
        Основной путь — FAISS; этот метод срабатывает при недоступности FAISS.
        Возвращает top-k записей (по одной на эмбеддинг), формат совместим со
        старой pgvector-реализацией: [{"user_id", "similarity"}].
        """
        vector = np.asarray(embedding, dtype=np.float32)

        norm = np.linalg.norm(vector)
        if norm == 0.0:
            raise ValueError("Invalid embedding vector")

        vector = vector / norm
        if vector.ndim != 1 or vector.shape[0] != 512:
            raise ValueError("Query embedding must be a 512-dim vector")

        matrix, user_ids = await self._aload_all_vectors_matrix("embedding_repo.find_top_k")
        if matrix is None:
            return []

        scores = matrix @ vector  # cosine, оба нормированы
        return self._topk_from_scores(scores, user_ids, k)

    async def find_top_k_batch(
        self,
        embeddings: Sequence[np.ndarray],
        k: int = 2,
    ) -> list[list[dict]]:
        """
        Batched SQL-fallback поиска: один decrypt-all, затем матрица запросов.
        Возвращает one top-k list per input embedding, сохраняя порядок.
        Формат совместим со старой pgvector-реализацией.
        """
        if not embeddings:
            return []

        queries: list[np.ndarray] = []
        for embedding in embeddings:
            vector = np.asarray(embedding, dtype=np.float32)

            norm = np.linalg.norm(vector)
            if norm == 0.0:
                raise ValueError("Invalid embedding vector")

            vector = vector / norm
            if vector.ndim != 1 or vector.shape[0] != 512:
                raise ValueError("Query embedding must be a 512-dim vector")

            queries.append(vector)

        qmat = np.stack(queries).astype(np.float32)  # (B, 512)

        matrix, user_ids = await self._aload_all_vectors_matrix(
            "embedding_repo.find_top_k_batch"
        )
        if matrix is None:
            return [[] for _ in embeddings]

        scores = qmat @ matrix.T  # (B, N)
        grouped: list[list[dict]] = []
        for b in range(len(queries)):
            grouped.append(self._topk_from_scores(scores[b], user_ids, k))
        return grouped

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
        Возвращает L2-нормализованные векторы эталонов пользователя (из encrypted).
        Используется для centroid-логики 1:1-верификации.
        """
        query = select(Embedding).where(Embedding.user_id == user_id)
        result = await timed_db_call(self.db.execute(query), "embedding_repo.get_user_vectors")
        rows = list(result.scalars().all())

        vectors: list[np.ndarray] = []
        for row in rows:
            decrypted = self._decrypt_record_embedding(row)
            if decrypted is not None:
                vectors.append(decrypted)

        return vectors

    async def get_all_vectors(self) -> list[dict]:
        """
        Аварийный fallback для search service: расшифровывает все эмбеддинги.
        Возвращает [{"user_id", "embedding": ndarray}] — формат совместим со
        старой реализацией (search_service сам считает cosine).
        Также используется при старте для построения FAISS-индекса.
        """
        query = select(
            Embedding.user_id,
            Embedding.encrypted_embedding,
            Embedding.id,
        )
        result = await timed_db_call(self.db.execute(query), "embedding_repo.get_all_vectors")
        rows = result.fetchall()

        items: list[dict] = []
        for row in rows:
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
