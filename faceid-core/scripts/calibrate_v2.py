import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.pipeline_v2 import FacePipelineV2
from app.services.calibration_service import CalibrationService


DATA = Path("tests/data")

pairs_same = [
    ("person1.jpg", "person1_2.jpg"),
]

pairs_diff = [
    ("person1.jpg", "person2.jpg"),
]

pipeline = FacePipelineV2()


def get_embedding(path: Path):
    result = pipeline.process(path.read_bytes())
    return result["embedding"]


def cosine(a, b):
    return float(np.dot(a, b))


scores = []
labels = []

# --- same person ---
for a, b in pairs_same:
    emb1 = get_embedding(DATA / a)
    emb2 = get_embedding(DATA / b)

    sim = cosine(emb1, emb2)

    print(f"SAME {a}-{b}: {sim:.4f}")

    scores.append(sim)
    labels.append(1)

# --- different person ---
for a, b in pairs_diff:
    emb1 = get_embedding(DATA / a)
    emb2 = get_embedding(DATA / b)

    sim = cosine(emb1, emb2)

    print(f"DIFF {a}-{b}: {sim:.4f}")

    scores.append(sim)
    labels.append(0)


result = CalibrationService.find_best_thresholds(scores, labels)

print("\n=== CALIBRATION RESULT ===")
print(result)
