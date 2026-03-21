# workers/celery_app.py

from celery import Celery
from app.core.config import settings


celery_app = Celery(
    "faceid_core",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)


celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

celery_app.conf.task_routes = {
    "app.workers.tasks.*": {"queue": "faceid"}
}
celery_app.conf.update(
    task_time_limit=25,
    task_soft_time_limit=15,
)
