# liveness_task.py - Задача проверки живости

from app.workers.celery_app import celery_app
from app.services.liveness_service import LivenessService


@celery_app.task
def run_liveness(image_bytes):

    service = LivenessService()

    result = service.check(image_bytes)

    return result