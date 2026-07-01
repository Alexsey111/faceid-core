# workers/tasks/liveness_task.py - liveness check task

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.core.config import settings
from app.ml.detection.retinaface_detector import RetinaFaceDetector
from app.ml.liveness.scoring import score_image_liveness
from app.ml.runtime import get_liveness_checker
from app.workers.celery_app import celery_app


def _decode(image_bytes: bytes) -> np.ndarray:
    """Декодирует байты в BGR uint8 кадр (без resize — checker сам делает кроп)."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image")
    return image


@celery_app.task(name="app.workers.tasks.run_liveness")
def run_liveness(image_bytes: bytes) -> dict[str, Any]:
    image = _decode(image_bytes)

    checker = get_liveness_checker()
    if checker is None:
        raise RuntimeError("Liveness model is not available")

    detector = RetinaFaceDetector(det_size=settings.RETINA_DET_SIZE_SMALL)
    result = score_image_liveness(
        image, detector, checker, settings.LIVENESS_THRESHOLD
    )

    return {
        "liveness": result["liveness"],
        "score": result["score"],
        "face_detected": result["face_detected"],
    }