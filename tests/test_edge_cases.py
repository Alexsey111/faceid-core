import numpy as np
import pytest
from pathlib import Path
from typing import cast

from app.ml.pipeline import FacePipeline
from app.ml.pipeline_runtime import get_pipeline
from app.services.verification_service import VerificationService
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.verification_repo import VerificationRepository


DATA_DIR = Path(__file__).parent / "data"


class DummyEmbeddingRepo(EmbeddingRepository):
    async def find_top_k(self, *args, **kwargs):
        return []

    async def get_all_vectors(self):
        return []


class DummyVerificationRepo(VerificationRepository):
    async def create_log(self, *args, **kwargs):
        pass


@pytest.mark.asyncio
async def test_empty_image_bytes():
    pipeline = get_pipeline()

    with pytest.raises(Exception):
        await pipeline.process_async(b"")


@pytest.mark.asyncio
async def test_invalid_image_bytes():
    pipeline = get_pipeline()

    invalid_bytes = b"not_an_image"

    with pytest.raises(Exception):
        await pipeline.process_async(invalid_bytes)


@pytest.mark.asyncio
async def test_no_face_detected():
    pipeline = get_pipeline()

    # полностью чёрное изображение
    blank = np.zeros((112, 112, 3), dtype=np.uint8).tobytes()

    with pytest.raises(Exception):
        await pipeline.process_async(blank)


@pytest.mark.asyncio
async def test_zero_embedding_handling():
    service = VerificationService(
        embedding_repo=DummyEmbeddingRepo(None),  # type: ignore
        verification_repo=DummyVerificationRepo(None),  # type: ignore
    )

    # подменим pipeline
    class FakePipeline:
        async def process_async(self, *args, **kwargs):
            return {
                "embedding": np.zeros(512, dtype=np.float32),
                "liveness": {}
            }

    service.pipeline = cast(FacePipeline, FakePipeline())

    result = await service.verify_face(b"fake")

    assert result["status"] == "no_match"
    assert result["similarity"] == 0.0


@pytest.mark.asyncio
async def test_corrupted_image_does_not_crash_service():
    service = VerificationService(
        embedding_repo=DummyEmbeddingRepo(None),  # type: ignore
        verification_repo=DummyVerificationRepo(None),  # type: ignore
    )

    result = await service.verify_face(b"corrupted_data")

    assert result["status"] in ["processing_failed", "no_match"]
