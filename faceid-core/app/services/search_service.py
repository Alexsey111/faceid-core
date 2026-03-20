# app/services/search_service.py

from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Union, cast

import hashlib
import json
import redis
import time
import numpy as np

from app.core.config import settings
from app.core.metrics import metrics
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
        self._redis = None
        if settings.FAISS_ENABLED and SearchService._faiss_index is None:
            SearchService._faiss_index = FaissIndex()
        if getattr(settings, "REDIS_ENABLED", False):
            try:
                self._redis = redis.Redis(
                    host=getattr(settings, "REDIS_HOST", "localhost"),
                    port=getattr(settings, "REDIS_PORT", 6379),
                    decode_responses=True
                )
            except Exception:
                self._redis = None

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
        start = time.time()
        try:
            query = np.asarray(embedding, dtype=np.float32)

            norm = np.linalg.norm(query)
            if norm != 0.0:
                query = query / norm

            cache_key = None
            if self._redis:
                try:
                    cache_key = f"faceid:search:{self._hash_embedding(query)}:{k}"
                    cached = self._redis.get(cache_key)
                    if cached:
                        cached_value = cast(Union[str, bytes, bytearray], cached)
                        if isinstance(cached_value, bytes):
                            cached_str = cached_value.decode()
                        else:
                            cached_str = cached_value
                        metrics.inc("redis_hit")
                        return json.loads(cached_str)
                except Exception:
                    metrics.inc("search_errors")
                    pass

            # 1. FAISS fast path
            if settings.FAISS_ENABLED and SearchService._faiss_index:
                try:
                    results = SearchService._faiss_index.search(query, k)
                    if results:
                        if self._redis and cache_key:
                            try:
                                self._redis.setex(cache_key, 3600, json.dumps(results))
                            except Exception:
                                pass
                        metrics.inc("faiss_hit")
                        return results
                except Exception:
                    metrics.inc("search_errors")
                    pass

            # 2. pgvector fast path
            if hasattr(self.embedding_repo, "find_top_k"):
                try:
                    results = await self.embedding_repo.find_top_k(query, k=k)
                    if results:
                        if self._redis and cache_key:
                            try:
                                self._redis.setex(cache_key, 3600, json.dumps(results))
                            except Exception:
                                pass
                        metrics.inc("db_fallback")
                        return results
                except Exception:
                    metrics.inc("search_errors")
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
                result = scored[:k]
                metrics.inc("db_fallback")
                if result and self._redis and cache_key:
                    try:
                        self._redis.setex(cache_key, 3600, json.dumps(result))
                    except Exception:
                        pass
                return result

            return []
        finally:
            metrics.observe("search_latency", time.time() - start)

    def _hash_embedding(self, embedding: np.ndarray) -> str:
        return hashlib.sha256(embedding.tobytes()).hexdigest()

    async def search_user_embeddings(
        self,
        user_id: int
    ):
        return await self.embedding_repo.get_user_vectors(user_id)
