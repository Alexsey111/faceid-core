from pathlib import Path


def resolve_liveness_model_path(models_dir: str | Path) -> Path | None:
    """
    Resolve the passive liveness model path with backward-compatible names.

    Preferred order:
    1. liveness.onnx
    2. antispoof.onnx
    """
    root = Path(models_dir)
    candidates = [
        root / "liveness.onnx",
        root / "antispoof.onnx",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None
