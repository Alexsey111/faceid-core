import numpy as np
from pathlib import Path

from app.ml.pipeline import FacePipeline


DATA_DIR = Path(__file__).parent / "data"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def test_same_person_similarity_high():
    pipeline = FacePipeline()

    img1 = (DATA_DIR / "person1.jpg").read_bytes()
    img2 = (DATA_DIR / "person1_2.jpg").read_bytes()

    r1 = pipeline.process(img1)
    r2 = pipeline.process(img2)

    e1 = r1["embedding"]
    e2 = r2["embedding"]

    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 / np.linalg.norm(e2)

    sim = cosine(e1, e2)

    assert sim > 0.60, f"Same person similarity too low: {sim}"


def test_different_person_similarity_low():
    pipeline = FacePipeline()

    img1 = (DATA_DIR / "person1.jpg").read_bytes()
    img2 = (DATA_DIR / "person2.jpg").read_bytes()

    r1 = pipeline.process(img1)
    r2 = pipeline.process(img2)

    e1 = r1["embedding"]
    e2 = r2["embedding"]

    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 / np.linalg.norm(e2)

    sim = cosine(e1, e2)

    assert sim < 0.4, f"Different person similarity too high: {sim}"


def test_embedding_is_normalized():
    pipeline = FacePipeline()

    img = (DATA_DIR / "person1.jpg").read_bytes()
    result = pipeline.process(img)

    e = result["embedding"]
    norm = np.linalg.norm(e)

    assert 0.99 < norm < 1.01, f"Embedding not normalized: norm={norm}"
