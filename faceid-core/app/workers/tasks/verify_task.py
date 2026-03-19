# workers/tasks/verify_task.py- Задача верификации

import numpy as np

from app.workers.celery_app import celery_app
from app.ml.pipeline import FacePipeline


pipeline = None


def get_pipeline():
    global pipeline

    if pipeline is None:
        pipeline = FacePipeline()

    return pipeline


@celery_app.task
def verify_face_task(image_bytes: bytes):

    """
    Async face verification task
    """

    pipeline = get_pipeline()

    result = pipeline.process(image_bytes)

    embedding = result["embedding"]

    return {
        "embedding": embedding.tolist(),
        "bbox": result["bbox"],
        "landmarks": result["landmarks"],
    }