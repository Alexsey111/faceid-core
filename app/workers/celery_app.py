# app\workers\celery_app.py

import logging
import os

from celery import Celery
from celery.signals import setup_logging as celery_setup_logging, worker_ready
from kombu import Queue
from prometheus_client import start_http_server
from app.core.config import settings
from app.core.logger import setup_logging

# НЕ вызываем setup_logging() на импорте: он делает root.handlers = [handler],
# что затёрло бы конфигурацию логирования Celery и ломало вывод при импорте
# модуля (в т.ч. в тестах/других worker-тасках). Вместо этого подключаемся к
# Celery-сигналу setup_logging — он срабатывает при старте worker'а до
# дефолтного Celery-hijack root-логгера, позволяя полностью заменить конфиг
# на наш JSON-логгер с BiometryRedactionFilter.
logger = logging.getLogger(__name__)
logging.getLogger("insightface").setLevel(logging.WARNING)
logging.getLogger("onnxruntime").setLevel(logging.WARNING)


@celery_setup_logging.connect
def _configure_logging(**_kwargs) -> None:
    """JSON-логгер с redaction для celery-worker'а (вместо Celery-дефолта)."""
    setup_logging()


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

_metrics_server_started = False


@worker_ready.connect
def _start_metrics_server(**_kwargs) -> None:
    global _metrics_server_started

    if _metrics_server_started:
        return

    port = int(os.getenv("PROMETHEUS_METRICS_PORT", "9100"))
    start_http_server(port)
    _metrics_server_started = True
    logger.info("prometheus_metrics_server_started port=%s", port)
    logger.info(
        "service_runtime_snapshot",
        extra=settings.service_runtime_snapshot("worker"),
    )
