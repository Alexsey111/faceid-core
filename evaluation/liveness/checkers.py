# evaluation/liveness/checkers.py — liveness-чекеры с явным контрактом preprocess/crop.
#
# Разные модели anti-spoofing имеют разные контракты (размер входа, нормализация,
# порядок классов, crop scale). Чтобы честно сравнивать модели в eval-harness,
# каждый чекер инкапсулирует свой контракт и exposes predict(image_bgr, bbox_xyxy).
#
# Контракт:
#   MiniFASNetChecker (yakhyo MiniFASNetV2/V1SE, CelebA-Spoof-trained):
#     - 80×80, BGR, 0-255 float32 (БЕЗ /255), NCHW
#     - 3 класса [print?/live/replay?], idx1 = Real (live)
#     - crop scale 2.7 (V2) / 4.0 (V1SE) — квадратный кроп с расширением, clamp в границы
# (legacy 128×128 2-класс liveness.onnx снят с production; ablation завершён — чекер удалён)

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

# crop_face_square/_softmax — общий кроп-контракт, живёт в app.* (используется
# И production-чекером OnnxLivenessChecker, И eval-harness'ом). Импорт из app.*
# корректен: run_eval.py вставляет git-root в sys.path перед импортом app.*.
from app.ml.liveness.crop import _softmax, crop_face_square

logger = logging.getLogger("liveness.checkers")


class MiniFASNetChecker:
    """yakhyo MiniFASNetV2/V1SE (CelebA-Spoof-trained, 3-класс, 80×80, BGR 0-255)."""

    def __init__(self, model_path: str | Path, crop_scale: float = 2.7, input_size: int = 80):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.input_size = int(shape[-1]) if shape and shape[-1] and not isinstance(shape[-1], str) else input_size
        self.crop_scale = float(crop_scale)
        self.real_idx = 1  # idx1 = Real (yakhyo конвенция)

    def predict(self, image_bgr: np.ndarray, bbox_xyxy) -> tuple[float, bool]:
        crop = crop_face_square(image_bgr, bbox_xyxy, self.crop_scale, self.input_size)
        if crop is None:
            return 0.0, False
        face = crop.astype(np.float32)            # 0-255, БЕЗ нормализации
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)
        logits = self.session.run(None, {self.input_name: face})[0]
        probs = _softmax(np.asarray(logits, dtype=np.float32))
        return float(probs[0, self.real_idx]), True


# реестр моделей: slug → (factory, дефолт crop_scale, описание)
MODEL_REGISTRY = {
    "yakhyo_v2": dict(
        factory=lambda p: MiniFASNetChecker(p, crop_scale=2.7, input_size=80),
        path_candidates=(
            # тот же файл, что находит production-resolver (MODELS_DIR/MiniFASNetV2_yakhyo.onnx)
            r"C:/Users/Worker/ai/faceid-core/models/MiniFASNetV2_yakhyo.onnx",
        ),
        crop_scale=2.7, desc="yakhyo MiniFASNetV2 (CelebA-Spoof, 80×80, 3-class idx1=real)",
    ),
    "yakhyo_v1se": dict(
        factory=lambda p: MiniFASNetChecker(p, crop_scale=4.0, input_size=80),
        path_candidates=(
            r"C:/Users/Worker/ai/faceid-core/models/liveness_candidates/MiniFASNetV1SE_yakhyo.onnx",
        ),
        crop_scale=4.0, desc="yakhyo MiniFASNetV1SE (CelebA-Spoof, 80×80, 3-class idx1=real)",
    ),
}


def resolve_model_path(slug: str) -> Path:
    for c in MODEL_REGISTRY[slug]["path_candidates"]:
        if Path(c).is_file():
            return Path(c)
    raise SystemExit(f"model file for '{slug}' not found in: {MODEL_REGISTRY[slug]['path_candidates']}")


def build_checker(slug: str):
    entry = MODEL_REGISTRY[slug]
    p = resolve_model_path(slug)
    logger.info("liveness model [%s]: %s", slug, p)
    return entry["factory"](p)