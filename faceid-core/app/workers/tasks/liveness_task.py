# workers/tasks/liveness_task.py - liveness check task

from __future__ import annotations

import cv2
import numpy as np

from app.ml.runtime import get_liveness_model
from app.workers.celery_app import celery_app


THRESHOLD = 0.7


def _preprocess(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image")

    image = cv2.resize(image, (128, 128))
    image = image.astype("float32") / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image


@celery_app.task(name="app.workers.tasks.run_liveness")
def run_liveness(image_bytes: bytes) -> dict[str, float | bool]:
    image = _preprocess(image_bytes)

    session = get_liveness_model()
    if session is None:
        raise RuntimeError("Liveness model is not available")

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image})
    score = float(outputs[0][0][1])

    return {
        "liveness": score > THRESHOLD,
        "score": score,
    }
