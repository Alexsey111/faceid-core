# app/services/search_service.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, List, Dict, cast

import hashlib
import json
import redis
import numpy as np
from time import perf_counter

from app.core.config import settings
from app.monitoring.metrics import (
    FAISS_HIT,
    REDIS_HIT,
    DB_FALLBACK,
    ERROR_COUNTER,
    REDIS_COMMAND_LATENCY_MS,
    SEARCH_BACKEND_COUNTER,
    SEARCH_LATENCY,
)
from app.services.faiss_index import FaissIndex

REDIS_POOL = redis.ConnectionPool(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    decode_responses=True,
    max_connections=50,
)

if TYPE_CHECKING:
    from app.db.repositories.embedding_repo import EmbeddingRepository


class SearchService:
    """
    Стратегия поиска:
    1. FAISS
    2. pgvector
    3. CPU fallback по всем векторам
    """

    _faiss_index: ClassVar[FaissIndex | None] = None

    def __init__(self, embedding_repo: "EmbeddingRepository"):
        self.embedding_repo = embedding_repo
        self._redis = None

        if settings.FAISS_ENABLED and SearchService._faiss_index is None:
            SearchService._faiss_index = FaissIndex()

        if getattr(settings, "REDIS_ENABLED", False):
            self._redis = redis.Redis(connection_pool=REDIS_POOL)

    def add_embedding(self, vector: np.ndarray, user_id: int) -> None:
        if settings.FAISS_ENABLED and SearchService._faiss_index is not None:
            try:
                SearchService._faiss_index.add_one(vector, user_id)
            except Exception:
                pass

    async def search_top_k(
        self,
        embedding: np.ndarray,
        k: int = 2
    ) -> List[Dict]:
        with SEARCH_LATENCY.time():
            try:
                sim_threshold = getattr(settings, "SIM_THRESHOLD", 0.5)
                query = np.asarray(embedding, dtype=np.float32)
                norm = np.linalg.norm(query)
                if norm != 0.0:
                    query = query / norm

                def filter_results(results: List[Dict]) -> List[Dict]:
                    return [
                        r for r in results
                        if r.get("similarity", 0.0) >= sim_threshold
                    ]

                cache_key = None
                if self._redis is not None:
                    try:
                        cache_key = f"faceid:search:{self._hash_embedding(query)}:{k}"
                        start = perf_counter()
                        cached = self._redis.get(cache_key)
                        REDIS_COMMAND_LATENCY_MS.labels(command="search_cache_get").observe(
                            (perf_counter() - start) * 1000.0
                        )
                        if cached is not None:
                            REDIS_HIT.labels(
                                endpoint="search_top_k",
                                result="hit",
                            ).inc()
                            SEARCH_BACKEND_COUNTER.labels(backend="redis").inc()
                            return json.loads(cast(str, cached))
                    except Exception as exc:
                        ERROR_COUNTER.labels(
                            stage="redis_cache",
                            error_type=type(exc).__name__,
                        ).inc()

                if settings.FAISS_ENABLED and SearchService._faiss_index is not None:
                    try:
                        results = SearchService._faiss_index.search(query, k)
                        if results:
                            if self._redis is not None and cache_key is not None:
                                try:
                                    start = perf_counter()
                                    self._redis.setex(cache_key, 3600, json.dumps(results))
                                    REDIS_COMMAND_LATENCY_MS.labels(command="search_cache_setex").observe(
                                        (perf_counter() - start) * 1000.0
                                    )
                                except Exception:
                                    pass
                            FAISS_HIT.labels(
                                endpoint="search_top_k",
                                result="hit",
                            ).inc()
                            SEARCH_BACKEND_COUNTER.labels(backend="faiss").inc()
                            filtered = filter_results(results)
                            if filtered:
                                return filtered
                    except Exception as exc:
                        ERROR_COUNTER.labels(
                            stage="faiss_search",
                            error_type=type(exc).__name__,
                        ).inc()

                if hasattr(self.embedding_repo, "find_top_k"):
                    try:
                        results = await self.embedding_repo.find_top_k(query, k=k)
                        if results:
                            if self._redis is not None and cache_key is not None:
                                try:
                                    start = perf_counter()
                                    self._redis.setex(cache_key, 3600, json.dumps(results))
                                    REDIS_COMMAND_LATENCY_MS.labels(command="search_cache_setex").observe(
                                        (perf_counter() - start) * 1000.0
                                    )
                                except Exception:
                                    pass
                            DB_FALLBACK.labels(
                                endpoint="search_top_k",
                                result="fallback",
                            ).inc()
                            SEARCH_BACKEND_COUNTER.labels(backend="db").inc()
                            if results:
                                return results
                    except Exception as exc:
                        ERROR_COUNTER.labels(
                            stage="db_search",
                            error_type=type(exc).__name__,
                        ).inc()

                if hasattr(self.embedding_repo, "get_all_vectors"):
                    items = await self.embedding_repo.get_all_vectors()
                    if not items:
                        return []

                    scored: list[dict] = []
                    for item in items:
                        vector = np.asarray(item["embedding"], dtype=np.float32)
                        v_norm = np.linalg.norm(vector)
                        if v_norm == 0.0:
                            continue

                        vector = vector / v_norm
                        similarity = float(np.dot(query, vector))
                        scored.append({
                            "user_id": item["user_id"],
                            "similarity": similarity,
                        })

                    scored.sort(key=lambda x: x["similarity"], reverse=True)
                    result = scored[:k]

                    DB_FALLBACK.labels(
                        endpoint="search_top_k",
                        result="fallback",
                    ).inc()
                    SEARCH_BACKEND_COUNTER.labels(backend="db").inc()

                    if result and self._redis is not None and cache_key is not None:
                        try:
                            start = perf_counter()
                            self._redis.setex(cache_key, 3600, json.dumps(result))
                            REDIS_COMMAND_LATENCY_MS.labels(command="search_cache_setex").observe(
                                (perf_counter() - start) * 1000.0
                            )
                        except Exception:
                            pass

                    return result

                return []

            except Exception as exc:
                ERROR_COUNTER.labels(
                    stage="search_top_k",
                    error_type=type(exc).__name__,
                ).inc()
                return []

    def _hash_embedding(self, embedding: np.ndarray) -> str:
        rounded = np.round(embedding, 5)
        return hashlib.sha256(rounded.tobytes()).hexdigest()

    async def invalidate_cache(self, user_id: int | None = None) -> None:
        if self._redis is None:
            return

        try:
            keys = list(self._redis.scan_iter("faceid:search:*"))
            if keys:
                self._redis.delete(*keys)
        except Exception:
            pass

    async def search_user_embeddings(self, user_id: int):
        return await self.embedding_repo.get_user_vectors(user_id)
