# app\services\verification_service_factory.py

from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.pipeline_v2 import FacePipelineV2
from app.services.verification_service import VerificationService
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.verification_repo import VerificationRepository
from app.services.search_service import SearchService



class PipelineProtocol(Protocol):
    def process(self, image_bytes: bytes) -> dict[str, Any]: ...


_pipeline: PipelineProtocol | None = None


def get_pipeline() -> PipelineProtocol:
    global _pipeline
    if _pipeline is None:
        _pipeline = FacePipelineV2()
        print(f"Using pipeline: {type(_pipeline).__name__}", flush=True)
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


def get_verification_service_without_pipeline(db: AsyncSession) -> VerificationService:
    embedding_repo = EmbeddingRepository(db)
    verification_repo = VerificationRepository(db)

    return VerificationService(
        embedding_repo=embedding_repo,
        verification_repo=verification_repo,
        search_service=SearchService(embedding_repo),
        load_pipeline=False,
    )
