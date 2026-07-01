from pathlib import Path


def resolve_liveness_model_path(models_dir: str | Path) -> Path | None:
    """
    Resolve the passive liveness model path.

    Единственный кандидат — yakhyo MiniFASNetV2 (CelebA-Spoof, 80×80, 3-класс
    idx1=real, BGR 0-255 без /255), AUC 0.97 на Anti-Spoofing Dataset.
    OnnxLivenessChecker требует 3-класс выход (guard). Если файл отсутствует —
    возвращаем None: pipeline_v2 / route / task логируют «liveness disabled»
    (graceful degradation) вместо падения.
    """
    root = Path(models_dir)
    candidates = [
        root / "MiniFASNetV2_yakhyo.onnx",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None
