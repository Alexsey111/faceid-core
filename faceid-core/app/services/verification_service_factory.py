from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.verification_service import VerificationService
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.verification_repo import VerificationRepository


_service: Optional[VerificationService] = None


def get_verification_service(db: AsyncSession) -> VerificationService:
    global _service
    if _service is None:
        _service = VerificationService(
            EmbeddingRepository(db),
            VerificationRepository(db)
        )
    return _service
