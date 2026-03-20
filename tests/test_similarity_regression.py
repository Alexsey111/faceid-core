import numpy as np
import pytest
from pathlib import Path

from app.ml.pipeline_runtime import get_pipeline


DATA_DIR = Path(__file__).parent / "data"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


@pytest.mark.asyncio
async def test_same_person_similarity_high():
    pipeline = get_pipeline()

    img1 = (DATA_DIR / "person1.jpg").read_bytes()
    img2 = (DATA_DIR / "person1_2.jpg").read_bytes()

    r1 = await pipeline.process_async(img1)
    r2 = await pipeline.process_async(img2)

    e1 = r1["embedding"]
    e2 = r2["embedding"]

    # нормализация (на случай если где-то сломается)
    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 / np.linalg.norm(e2)

    sim = cosine(e1, e2)

    # same person
    assert sim > 0.65, f"Same person similarity too low: {sim}"


@pytest.mark.asyncio
async def test_different_person_similarity_low():
    pipeline = get_pipeline()

    img1 = (DATA_DIR / "person1.jpg").read_bytes()
    img2 = (DATA_DIR / "person2.jpg").read_bytes()

    r1 = await pipeline.process_async(img1)
    r2 = await pipeline.process_async(img2)

    e1 = r1["embedding"]
    e2 = r2["embedding"]

    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 / np.linalg.norm(e2)

    sim = cosine(e1, e2)

    # different person
    assert sim < 0.4, f"Different person similarity too high: {sim}"


@pytest.mark.asyncio
async def test_embedding_is_normalized():
    pipeline = get_pipeline()

    img = (DATA_DIR / "person1.jpg").read_bytes()
    result = await pipeline.process_async(img)

    e = result["embedding"]
    norm = np.linalg.norm(e)

    assert 0.99 < norm < 1.01, f"Embedding not normalized: norm={norm}"
