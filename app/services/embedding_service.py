# app/services/embedding_service.py - Сервис эмбеддингов

import asyncio
from typing import Dict, Any, Optional

import numpy as np
from app.core.vector import l2_normalize
from app.services.verification_service_factory import get_pipeline
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.user_repo import UserRepository
from app.services.search_service import SearchService


class EmbeddingService:

    def __init__(self, embedding_repo: EmbeddingRepository, user_repo: Optional[UserRepository] = None):

        # Pipeline — через общий factory (FacePipelineV2, единый синглтон с
        # verification_service). Раньше хардкодился FacePipeline (V1, удалён),
        # чей process() НЕ возвращал "status" → проверка status!="ok" всегда
        # падала с enroll_failed → /upload и /update-reference были сломаны.
        self.pipeline = get_pipeline()
        self.embedding_repo = embedding_repo
        self.user_repo = user_repo
        self.search_service = SearchService(embedding_repo)

    async def enroll_face(
        self,
        user_id: int | str,
        image_bytes: bytes
    ) -> Dict[str, Any]:
        if user_id is None:
            raise ValueError("user_id is required")

        external_id = str(user_id)

        # Create user if not exists, get internal integer ID for DB relations.
        internal_user_id: int
        if self.user_repo:
            user = await self.user_repo.get_or_create(external_id)
            internal_user_id = user.id
        else:
            # Legacy/tests path: no repo → try integer fallback.
            try:
                internal_user_id = int(user_id)
            except (ValueError, TypeError):
                internal_user_id = user_id  # type: ignore[assignment]

        # pipeline.process — тяжёлый ML-инференс (детект+encode). В async-контексте
        # /upload обернуть в executor, иначе блокируется event loop (audit E1 —
        # parity с /verify_face → _verify_face_impl, который тоже в to_thread).
        result = await asyncio.to_thread(self.pipeline.process, image_bytes)

        # Pipeline может вернуть не-"ok" status (quality_reject/retry/no_face/spoof) —
        # в этих случаях ключа "embedding" нет. Раньше падало KeyError → HTTP 500.
        # Поднимаем ValueError с reason → HTTP 400 (как в /verify), клиент получает
        # структурированный отказ вместо 500.
        if result.get("status") != "ok":
            reason = result.get("quality_reason") or result.get("status")
            raise ValueError(f"enroll_failed: {reason}")

        embedding: np.ndarray = result["embedding"]
        # normalize embedding (защитно — энкодер уже нормирует, но инвариант
        # может быть нарушен; единая l2_normalize из app.core.vector)
        embedding = l2_normalize(embedding)
        embedding = embedding.tolist()

        # Replace-семантика: один эталон на пользователя. Перед create удаляем
        # старые эмбеддинги этого user (как /update-reference). Раньше /upload
        # копил по записи на каждый вызов без лимита (audit upload-gaps).
        await self.embedding_repo.delete_by_user_id(internal_user_id)

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

        # In-process FAISS-обновление (audit E3). Раньше для обновления индекса на
        # /upload звали Celery-таску (delay) в try/except: pass — при ОТКЛЮЧЁННОМ
        # Celery (default-deploy) это молча падало → FAISS-индекс НЕ обновлялся →
        # устаревал → ложные no_match. Celery-путь удалён; теперь прямой
        # синхронный вызов: SearchService.add_embedding сам обёрнут в try/except
        # (не ломает enroll), add_one на FAISS — C-операция (GIL-release), микросекунды.
        vector = np.asarray(embedding, dtype=np.float32)
        self.search_service.add_embedding(vector, internal_user_id)

        return {
            "embedding_id": record.id,
            "user_id": user_id
        }
