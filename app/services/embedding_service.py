# app/services/embedding_service.py - Сервис эмбеддингов

from typing import Dict, Any, Optional

import numpy as np
from app.ml.pipeline import FacePipeline
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.user_repo import UserRepository
from app.services.search_service import SearchService
from app.workers.tasks.faiss_tasks import add_embedding_task


class EmbeddingService:

    def __init__(self, embedding_repo: EmbeddingRepository, user_repo: Optional[UserRepository] = None):

        self.pipeline = FacePipeline()
        self.embedding_repo = embedding_repo
        self.user_repo = user_repo
        self.search_service = SearchService(embedding_repo)

    async def enroll_face(
        self,
        user_id: int,
        image_bytes: bytes
    ) -> Dict[str, Any]:
        if user_id is None:
            raise ValueError("user_id is required")

        # Create user if not exists, get internal ID
        internal_user_id: int = user_id
        if self.user_repo:
            user = await self.user_repo.get_or_create(str(user_id))
            internal_user_id = getattr(user, 'id', user_id)

        result = self.pipeline.process(image_bytes)

        # Pipeline может вернуть не-"ok" status (quality_reject/retry/no_face/spoof) —
        # в этих случаях ключа "embedding" нет. Раньше падало KeyError → HTTP 500.
        # Поднимаем ValueError с reason → HTTP 400 (как в /verify), клиент получает
        # структурированный отказ вместо 500.
        if result.get("status") != "ok":
            reason = result.get("quality_reason") or result.get("status")
            raise ValueError(f"enroll_failed: {reason}")

        embedding: np.ndarray = result["embedding"]
        # normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        embedding = embedding.tolist()

        record = await self.embedding_repo.create_embedding(
            user_id=internal_user_id,
            embedding=embedding
        )

        # CRITICAL: invalidate Redis cache after enrollment
        try:
            await self.search_service.invalidate_cache()
        except Exception:
            # не ломаем enroll
            pass

        try:
            vector = np.asarray(embedding, dtype=np.float32)
            add_embedding_task.delay(vector.tolist(), internal_user_id)
        except Exception:
            # не ломаем enroll
            pass

        return {
            "embedding_id": record.id,
            "user_id": user_id
        }
