# app/core/vector.py — L2-нормализация векторов (единая реализация).
#
# Инвариант: эмбеддинг хранится и сравнивается как unit-вектор (cosine = dot
# product). Нормируется один раз в энкодере (onnx_arcface_encoder._l2_normalize);
# нижележащие слои (сервисы, репо) нормируют защитно — на случай нарушения
# инварианта. Guard-логика (raise на zero / skip / continue) остаётся в caller:
# здесь только базовое v/||v|| без guard (zero → NaN/inf, caller проверяет сам).

import numpy as np


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Базовая L2-нормализация: v / ||v||. Без guard от zero-вектора."""
    return v / np.linalg.norm(v)