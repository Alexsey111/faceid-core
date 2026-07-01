# app/ml/liveness/scoring.py — общий scorer liveness для standalone-путей
# (POST /liveness route и celery task run_liveness).
#
# Зеркалит evaluation.liveness.predict.score_frame: downscale (ImagePreprocessor)
# → RetinaFaceDetector top1 → checker.predict(image, bbox) → решение по порогу.
# Используется там, где на вход приходят «сырые» байты/кадр без bbox (нет pipeline):
# детекция лица обязательна — модель обучена на кропах лиц, кормить весь кадр
# нельзя (результат не совпадёт с измеренным AUC 0.97).

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.ml.detection.retinaface_detector import RetinaFaceDetector
from app.ml.preprocessing.image_preprocessor import ImagePreprocessor

logger = logging.getLogger("liveness.scoring")


def score_image_liveness(
    image_bgr: np.ndarray,
    detector: RetinaFaceDetector,
    checker: Any,
    threshold: float,
) -> dict[str, Any]:
    """
    Оценка liveness по полному кадру: downscale → детекция top1 → checker.predict.

    Args:
        image_bgr: BGR uint8 (H, W, 3) — полный кадр (не кроп).
        detector: RetinaFaceDetector (детектор должен быть тем же, что в pipeline,
            чтобы bbox-контракт совпадал с тем, на чём обучен checker).
        checker: OnnxLivenessChecker (или любой LivenessChecker Protocol с
            predict(image_bgr, bbox_xyxy) -> (real_score, ok)).
        threshold: порог решения (settings.LIVENESS_THRESHOLD).

    Returns:
        {"liveness": bool, "score": float, "face_detected": bool}.
        Если лицо не найдено или кроп пуст — liveness=False, score=0.0,
        face_detected=False.
    """
    preprocessor = ImagePreprocessor()
    img = preprocessor.process_image(image_bgr)

    faces = detector.detect(img) or []
    if not faces:
        return {"liveness": False, "score": 0.0, "face_detected": False}

    top = faces[0]
    real_score, ok = checker.predict(img, top["bbox"])
    if not ok:
        return {"liveness": False, "score": 0.0, "face_detected": False}

    return {
        "liveness": bool(real_score >= threshold),
        "score": float(real_score),
        "face_detected": True,
    }