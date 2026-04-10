from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "generated_hard"
BASE_IMAGES = [
    ROOT / "tests" / "data" / "person1.jpg",
    ROOT / "tests" / "data" / "person1_2.jpg",
    ROOT / "tests" / "data" / "person2.jpg",
]


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def _jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else image


def _apply_variant(image: np.ndarray, idx: int) -> np.ndarray:
    h, w = image.shape[:2]
    variant = image.copy()

    # Deterministic but varied transforms.
    if idx % 5 == 0:
        variant = cv2.GaussianBlur(variant, (9, 9), 0)
    elif idx % 5 == 1:
        variant = cv2.resize(variant, (max(64, w // 3), max(64, h // 3)), interpolation=cv2.INTER_AREA)
        variant = cv2.resize(variant, (w, h), interpolation=cv2.INTER_LINEAR)
    elif idx % 5 == 2:
        angle = (-12 + (idx % 7) * 4)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        variant = cv2.warpAffine(variant, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    elif idx % 5 == 3:
        # Partial occlusion.
        x1 = int(w * 0.22)
        y1 = int(h * 0.2)
        x2 = int(w * 0.78)
        y2 = int(h * 0.34)
        cv2.rectangle(variant, (x1, y1), (x2, y2), (0, 0, 0), -1)
    else:
        # Low-light + noise + jpeg compression.
        variant = cv2.convertScaleAbs(variant, alpha=0.55, beta=-18)
        noise = np.random.default_rng(idx).normal(0, 10, size=variant.shape).astype(np.int16)
        variant = np.clip(variant.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        variant = _jpeg_roundtrip(variant, quality=28)

    # Add a final mild crop/pad effect to make detection less trivial.
    pad = max(2, min(h, w) // 24)
    variant = variant[pad:h - pad, pad:w - pad]
    variant = cv2.resize(variant, (w, h), interpolation=cv2.INTER_LINEAR)
    return variant


def generate(out_dir: Path, count: int = 12) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for idx in range(count):
        base = BASE_IMAGES[idx % len(BASE_IMAGES)]
        image = _read_image(base)
        hard = _apply_variant(image, idx)
        out_path = out_dir / f"hard_{idx + 1:02d}.jpg"
        ok = cv2.imwrite(str(out_path), hard, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError(f"Failed to write {out_path}")
        outputs.append(out_path)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()

    outputs = generate(args.out_dir, args.count)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
