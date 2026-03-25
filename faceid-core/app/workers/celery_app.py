# faceid-core\app\workers\celery_app.py

import logging

from celery import Celery
from kombu import Queue
from app.core.config import settings

logging.getLogger("insightface").setLevel(logging.WARNING)
logging.getLogger("onnxruntime").setLevel(logging.WARNING)


celery_app = Celery(
    "faceid_core",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.workers.tasks.verify_task"],
)


celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_default_queue="verify_fast",
    task_default_priority=9,
    task_queue_max_priority=10,
    task_queues=(
        Queue("verify_fast"),
        Queue("verify_heavy"),
    ),
)
celery_app.conf.broker_transport_options = {
    "queue_order_strategy": "priority",
    "priority_steps": list(range(10)),
}
celery_app.conf.worker_prefetch_multiplier = 1

celery_app.conf.task_routes = {
    "app.workers.tasks.verify_job": {"queue": "verify_fast"},
}
celery_app.conf.update(
    task_time_limit=25,
    task_soft_time_limit=15,
)
