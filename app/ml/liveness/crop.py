# app/ml/liveness/crop.py — общий кроп-контракт для liveness-моделей.
#
# `crop_face_square` — квадратный кроп с центром лица, расширенный до `scale`×
# стороны bbox, clamp в границы изображения, resize в out_size×out_size.
# Совпадает с yakhyo AntiSpoofingONNX._crop_face и контрактом, на котором
# обучены MiniFASNetV2/V1SE (CelebA-Spoof). Используется И production-чекером
# (OnnxLivenessChecker), И eval-harness'ом (evaluation.liveness.checkers),
# чтобы prod и eval шли по идентичному пути препроцессинга.
#
# Модуль общий (app.*), чтобы не было дублирования и обратной зависимости
# app → evaluation (которая была бы архитектурно неправильной).

from __future__ import annotations

import cv2
import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    """Стабильный softmax по последней оси (ожидается 2D (N, C))."""
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def crop_face_square(
    image: np.ndarray,
    bbox_xyxy,
    scale: float,
    out_size: int,
) -> np.ndarray | None:
    """
    Silent-Face/yakhyo crop: квадратный кроп с центром лица, расширенный до `scale`×
    стороны bbox, clamp в границы изображения, resize в out_size×out_size.

    Args:
        image: BGR uint8 (H, W, 3).
        bbox_xyxy: (x1, y1, x2, y2) — bbox детектора (в координатах image).
        scale: во сколько раз расширить сторону bbox (2.7 для yakhyo MiniFASNetV2,
            4.0 для V1SE). Фактический scale ограничен размером изображения.
        out_size: сторона выхода (80 для yakhyo MiniFASNet).

    Returns:
        out_size×out_size BGR uint8, или None если bbox невалиден / кроп пуст.
    """
    src_h, src_w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None
    # фактический scale ограничен размером изображения и запрошенным scale
    s = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
    new_w = box_w * s
    new_h = box_h * s
    cx = x1 + box_w / 2
    cy = y1 + box_h / 2
    nx1 = max(0, int(cx - new_w / 2))
    ny1 = max(0, int(cy - new_h / 2))
    nx2 = min(src_w - 1, int(cx + new_w / 2))
    ny2 = min(src_h - 1, int(cy + new_h / 2))
    cropped = image[ny1:ny2 + 1, nx1:nx2 + 1]
    if cropped.size == 0:
        return None
    return cv2.resize(cropped, (out_size, out_size))