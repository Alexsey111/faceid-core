# faceid-core\app\services\verification_service_factory.py

from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.pipeline import FacePipeline
from app.services.verification_service import VerificationService
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.verification_repo import VerificationRepository
from app.services.search_service import SearchService


_pipeline: FacePipeline | None = None


def get_pipeline() -> FacePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = FacePipeline()
    return _pipeline


def get_verification_service(db: AsyncSession) -> VerificationService:
    embedding_repo = EmbeddingRepository(db)
    verification_repo = VerificationRepository(db)

    return VerificationService(
        embedding_repo=embedding_repo,
        verification_repo=verification_repo,
        search_service=SearchService(embedding_repo),
        pipeline=get_pipeline(),
    )
