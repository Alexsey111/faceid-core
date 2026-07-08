import numpy as np
import pytest

from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.verification_repo import VerificationRepository
from app.ml.pipeline_v2 import FacePipelineV2
from app.services.verification_service import VerificationService


class DummyEmbeddingRepo(EmbeddingRepository):
    async def find_top_k(self, *args, **kwargs):
        return []

    async def get_all_vectors(self):
        return []


class DummyVerificationRepo(VerificationRepository):
    async def create_log(self, *args, **kwargs):
        pass


def test_empty_image_bytes():
    pipeline = FacePipelineV2()

    with pytest.raises(Exception):
        pipeline.process(b"")


def test_invalid_image_bytes():
    pipeline = FacePipelineV2()

    with pytest.raises(Exception):
        pipeline.process(b"not_an_image")


def test_no_face_detected():
    pipeline = FacePipelineV2()
    blank = np.zeros((112, 112, 3), dtype=np.uint8).tobytes()

    with pytest.raises(Exception):
        pipeline.process(blank)


@pytest.mark.asyncio
async def test_zero_embedding_handling():
    service = VerificationService(
        embedding_repo=DummyEmbeddingRepo(None),  # type: ignore[arg-type]
        verification_repo=DummyVerificationRepo(None),  # type: ignore[arg-type]
    )

    class FakePipeline:
        def process(self, *args, **kwargs):
            return {
                "embedding": np.zeros(512, dtype=np.float32),
                "liveness": {},
            }

    service.pipeline = FakePipeline()

    result = await service.verify_face(b"fake")

    assert result["status"] == "no_match"
    assert result["similarity"] == 0.0


@pytest.mark.asyncio
async def test_corrupted_image_does_not_crash_service():
    service = VerificationService(
        embedding_repo=DummyEmbeddingRepo(None),  # type: ignore[arg-type]
        verification_repo=DummyVerificationRepo(None),  # type: ignore[arg-type]
    )

    result = await service.verify_face(b"corrupted_data")

    assert result["status"] in ["processing_failed", "no_match"]
