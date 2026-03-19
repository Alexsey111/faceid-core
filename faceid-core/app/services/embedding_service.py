# app/services/embedding_service.py - Сервис эмбеддингов

import asyncio
from typing import Dict, Any, Optional
import numpy as np
from app.ml.pipeline_runtime import get_pipeline
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.user_repo import UserRepository


class EmbeddingService:

    def __init__(self, embedding_repo: EmbeddingRepository, user_repo: Optional[UserRepository] = None):

        self.pipeline = get_pipeline()
        self.embedding_repo = embedding_repo
        self.user_repo = user_repo

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

        result = await self.pipeline.process_async(image_bytes)

        embedding: np.ndarray = result["embedding"]
        # normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        embedding = embedding.tolist()

        print("DEBUG embedding type:", type(embedding))
        if isinstance(embedding, (bytes, bytearray)):
            print("❌ EMBEDDING IS BYTES (ERROR)")
        else:
            print("✅ EMBEDDING OK (vector)")
        print("DEBUG sample:", embedding[:5] if hasattr(embedding, "__getitem__") else embedding)

        record = await self.embedding_repo.create_embedding(
            user_id=internal_user_id,
            embedding=embedding
        )

        return {
            "embedding_id": record.id,
            "user_id": user_id
        }
