from __future__ import annotations

import base64
import sys
from pathlib import Path

import requests


def load_image_b64(image_path: Path) -> tuple[str, str]:
    suffixes = "".join(image_path.suffixes).lower()

    if suffixes.endswith(".b64.txt"):
        return image_path.read_text(encoding="utf-8").strip(), "text"

    if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        image_bytes = image_path.read_bytes()
        return base64.b64encode(image_bytes).decode("ascii"), "binary"

    raise ValueError(f"Unsupported image file type: {image_path}")


def main() -> int:
    image_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/data/person1.jpg")
    url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000/verify_async"

    image_b64, mode = load_image_b64(image_arg)
    payload = {
        "image_b64": image_b64,
        "user_id": None,
        "require_liveness": False,
    }

    print(f"image={image_arg}")
    print(f"mode={mode}")
    print(f"image_b64_len={len(image_b64)}")

    try:
        resp = requests.post(url, json=payload, timeout=30)
    except requests.RequestException as exc:
        print(f"request_failed={exc.__class__.__name__}: {exc}")
        return 1

    print(resp.status_code)
    print(resp.text[:1000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
