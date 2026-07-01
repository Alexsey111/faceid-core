# evaluation/liveness/predict.py — scorer кадра: детекция → checker.predict(crop).
#
# Универсален относительно контракта модели: checker сам делает кроп+preprocess
# (см. evaluation.liveness.checkers). score_frame лишь даунскейлит кадр,
# детектит top-1 лицо и делегирует чекеру.
#
# ImagePreprocessor.process_image (downscale ≤480) → RetinaFaceDetector top1
# → checker.predict(image, bbox) → (real_score, ok). no-face → skip.

from __future__ import annotations

import logging
from typing import Optional, Protocol

import numpy as np

from app.ml.detection.retinaface_detector import RetinaFaceDetector
from app.ml.preprocessing.image_preprocessor import ImagePreprocessor

logger = logging.getLogger("liveness.predict")


class LivenessChecker(Protocol):
    def predict(self, image_bgr: np.ndarray, bbox_xyxy) -> tuple[float, bool]:
        """(real_score, ok). real_score — softmax[idx_real], ok=False если кроп пуст."""
        ...


def score_frame(
    image_bgr: np.ndarray,
    detector: RetinaFaceDetector,
    checker: LivenessChecker,
    preprocessor: Optional[ImagePreprocessor] = None,
) -> tuple[Optional[float], bool]:
    """
    Returns (real_score, face_detected).
      real_score = softmax[idx_real] (чем выше, тем «живее»); None если лицо не найдено
                   или чекер не смог сделать кроп.
      face_detected = True если получен валидный real_score.
    """
    if preprocessor is None:
        preprocessor = ImagePreprocessor()

    img = preprocessor.process_image(image_bgr)
    faces = detector.detect(img)
    if not faces:
        return None, False

    top = faces[0]
    bbox = top["bbox"]
    real_score, ok = checker.predict(img, bbox)
    if not ok:
        return None, False
    return float(real_score), True