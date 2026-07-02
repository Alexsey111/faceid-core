# app/ml/liveness/landmarks.py — 106-точечные 2D face landmarks (2d106det)
# для active challenge liveness.
#
# Зачем: SCRFD/RetinaFace даёт только 5 точек (центры глаз, нос, углы рта) —
# этого НЕ хватает для EAR (blink нужен контур глаза из 6 точек) и для надёжного
# solvePnP yaw/pitch (turn/nod). 2d106det даёт 106 2D-точек (контуры глаз, бровей,
# рта, овала) — тот же ONNX-runtime, что и остальной стек, без нового тяжёлого dep.
#
# Обёртка над insightface.model_zoo.landmark.Landmark: bbox лица → (106,2) abs-px.
# Сессию (с провайдерами CUDA/CPU) создаёт runtime.get_landmarker_106 (@lru_cache).
from __future__ import annotations

import numpy as np


class _FaceLike:
    """Упрощённый аналог insight Face для insightface Landmark.get.

    insightface.model_zoo.landmark.Landmark.get читает ``face.bbox`` и пишет
    результат в ``face["landmark_2d_106"]``. SimpleNamespace не поддерживает
    __setitem__, поэтому нужен тонкий объект с обоими интерфейсами.
    """

    __slots__ = ("bbox", "_store")

    def __init__(self, bbox: list[float] | np.ndarray) -> None:
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self._store: dict[str, np.ndarray] = {}

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        self._store[key] = value

    def __getitem__(self, key: str) -> np.ndarray:
        return self._store[key]


class Landmarker106:
    """106-pt 2D landmarks через 2d106det.onnx.

    Сессия передаётся извне (runtime.get_landmarker_106) — чтобы переиспользовать
    provider-fallback (CUDA → CPU) и session-options из settings.
    """

    TASK_KEY = "landmark_2d_106"

    def __init__(self, model_path: str, session: "object | None" = None) -> None:
        from insightface.model_zoo.landmark import Landmark

        # session уже создан с нужными providers; Landmark при session=None
        # создаст свой (default providers) — нам это не нужно на production.
        self._lm = Landmark(model_path, session=session)

    def get(self, img: np.ndarray, bbox: list[float] | np.ndarray) -> np.ndarray | None:
        """Вернуть (106, 2) float32 landmarks в абсолютных пикселях или None.

        Args:
            img: BGR uint8, H×W×3.
            bbox: xyxy координаты лица (детект SCRFD/RetinaFace).
        """
        face = _FaceLike(bbox)
        try:
            self._lm.get(img, face)
        except Exception:
            # детект дал странный bbox / вырожденный кроп — не валить стрим
            return None
        lm = face._store.get(self.TASK_KEY)
        if lm is None:
            return None
        return np.asarray(lm, dtype=np.float32)