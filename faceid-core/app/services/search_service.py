# app/services/search_service.py

from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict

import numpy as np

from app.core.config import settings
from app.services.faiss_index import FaissIndex

if TYPE_CHECKING:
    from app.db.repositories.embedding_repo import EmbeddingRepository


class SearchService:
    """
    Стратегия поиска:
    1. FAISS
    2. pgvector
    3. CPU fallback по всем векторам
    """

    _faiss_index: FaissIndex | None = None

    def __init__(self, embedding_repo: "EmbeddingRepository"):
        self.embedding_repo = embedding_repo
        if settings.FAISS_ENABLED and SearchService._faiss_index is None:
            SearchService._faiss_index = FaissIndex()

    def add_embedding(self, vector: np.ndarray, user_id: int) -> None:
        if settings.FAISS_ENABLED and SearchService._faiss_index:
            try:
                SearchService._faiss_index.add_one(vector, user_id)
            except Exception:
                # Не ломаем основной поток enroll
                pass

    async def search_top_k(
        self,
        embedding: np.ndarray,
        k: int = 2
    ) -> List[Dict]:
        query = np.asarray(embedding, dtype=np.float32)

        # 1. FAISS fast path
        if settings.FAISS_ENABLED and SearchService._faiss_index:
            try:
                results = SearchService._faiss_index.search(query, k)
                if results:
                    return results
            except Exception:
                pass

        # 2. pgvector fast path
        if hasattr(self.embedding_repo, "find_top_k"):
            try:
                results = await self.embedding_repo.find_top_k(query, k=k)
                if results:
                    return results
            except Exception:
                pass

        # 3. CPU fallback
        if hasattr(self.embedding_repo, "get_all_vectors"):
            items = await self.embedding_repo.get_all_vectors()
            if not items:
                return []

            query_norm = np.linalg.norm(query)
            if query_norm == 0.0:
                return []

            scored: list[dict] = []
            for item in items:
                vector = np.asarray(item["embedding"], dtype=np.float32)
                norm = np.linalg.norm(vector)
                if norm == 0.0:
                    continue

                similarity = float(np.dot(query, vector) / (query_norm * norm))
                scored.append({
                    "user_id": item["user_id"],
                    "similarity": similarity,
                })

            scored.sort(key=lambda x: x["similarity"], reverse=True)
            return scored[:k]

        return []

    async def search_user_embeddings(
        self,
        user_id: int
    ):
        return await self.embedding_repo.get_user_vectors(user_id)