from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "faceid",
    broker=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0",
    backend=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/1",
)

celery_app.conf.task_routes = {
    "app.tasks.*": {"queue": "faceid"}
}