import random
from pathlib import Path
from itertools import combinations
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "faceid-core"
sys.path.insert(0, str(APP_ROOT))

from app.ml.pipeline_v2 import FacePipelineV2
from app.services.calibration_service import CalibrationService


DATASET = PROJECT_ROOT / "tests" / "data_extended"

# ограничение diff пар (чтобы не было миллиона)
MAX_DIFF_PAIRS = 200


pipeline = FacePipelineV2()


def get_embedding(path: Path):
    try:
        result = pipeline.process(path.read_bytes())
        return result["embedding"]
    except ValueError as exc:
        print(f"SKIP {path.name}: {exc}")
        return None


def cosine(a, b):
    return float(np.dot(a, b))


def load_dataset():
    persons = {}

    for person_dir in DATASET.iterdir():
        if not person_dir.is_dir():
            continue

        images = list(person_dir.glob("*.*"))

        if len(images) < 2:
            continue

        persons[person_dir.name] = images

    return persons


def generate_same_pairs(persons):
    pairs = []

    for person, images in persons.items():
        for a, b in combinations(images, 2):
            pairs.append((a, b, 1))

    return pairs


def generate_diff_pairs(persons):
    pairs = []

    person_names = list(persons.keys())

    for i in range(len(person_names)):
        for j in range(i + 1, len(person_names)):
            p1 = person_names[i]
            p2 = person_names[j]

            for img1 in persons[p1]:
                for img2 in persons[p2]:
                    pairs.append((img1, img2, 0))

    # ограничиваем
    random.shuffle(pairs)
    return pairs[:MAX_DIFF_PAIRS]


def main():
    persons = load_dataset()

    print(f"Persons: {len(persons)}")

    same_pairs = generate_same_pairs(persons)
    diff_pairs = generate_diff_pairs(persons)

    print(f"Same pairs: {len(same_pairs)}")
    print(f"Diff pairs: {len(diff_pairs)}")

    scores = []
    labels = []

    # --- SAME ---
    print("\nProcessing SAME...")
    for a, b, label in same_pairs:
        emb1 = get_embedding(a)
        emb2 = get_embedding(b)

        if emb1 is None or emb2 is None:
            continue

        sim = cosine(emb1, emb2)

        scores.append(sim)
        labels.append(label)

    # --- DIFF ---
    print("\nProcessing DIFF...")
    for a, b, label in diff_pairs:
        emb1 = get_embedding(a)
        emb2 = get_embedding(b)

        if emb1 is None or emb2 is None:
            continue

        sim = cosine(emb1, emb2)

        scores.append(sim)
        labels.append(label)

    scores = np.array(scores)

    print("\n=== STATS ===")
    same_scores = scores[np.array(labels) == 1]
    diff_scores = scores[np.array(labels) == 0]

    print(f"SAME mean: {same_scores.mean():.4f}")
    print(f"SAME min/max: {same_scores.min():.4f} / {same_scores.max():.4f}")

    print(f"DIFF mean: {diff_scores.mean():.4f}")
    print(f"DIFF min/max: {diff_scores.min():.4f} / {diff_scores.max():.4f}")

    # --- calibration ---
    result = CalibrationService.find_best_thresholds(scores.tolist(), labels)

    print("\n=== CALIBRATION RESULT ===")
    print(result)


if __name__ == "__main__":
    main()
