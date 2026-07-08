# app\services\backpressure.py
from __future__ import annotations

import logging
import random

from app.core.config import settings
from app.infrastructure.redis_client import redis_client

IN_SYSTEM_KEY = "faceid:in_system"
QUEUE_DELAY_MS_KEY = "faceid:queue_delay_ms"
logger = logging.getLogger(__name__)


def current_active_requests() -> int:
    try:
        raw_value = redis_client.get(IN_SYSTEM_KEY)
        return max(0, int(raw_value or 0))
    except Exception:
        return 0


def get_system_load() -> float:
    try:
        raw_value = redis_client.get(QUEUE_DELAY_MS_KEY)
        return float(raw_value or 0.0)
    except Exception:
        return 0.0


def get_backpressure_mode(queue_delay_ms: float) -> str:
    if queue_delay_ms < 500:
        return "normal"
    elif queue_delay_ms < 1500:
        return "degrade"
    else:
        return "shed"


def should_drop_request(mode: str) -> bool:
    if mode == "shed":
        return random.random() < 0.5
    return False


def estimate_queue_delay_ms(in_system: int) -> float:
    active_tasks = max(0, in_system)
    throughput_per_sec = max(0.001, float(settings.ASYNC_THROUGHPUT_PER_SEC))
    return float((active_tasks / throughput_per_sec) * 1000.0)


def try_reserve_slot(max_queue_delay_ms: float = 2000.0) -> bool:
    in_system = redis_client.incr(IN_SYSTEM_KEY)
    redis_client.expire(IN_SYSTEM_KEY, 60)
    estimated_queue_delay_ms = estimate_queue_delay_ms(in_system)
    hard_limit = settings.MAX_ACTIVE_TASKS
    logger.info(
        "[BP] try_reserve in_system=%s hard_limit=%s throughput_per_sec=%.3f estimated_queue_delay_ms=%.3f max_queue_delay_ms=%.3f",
        in_system,
        hard_limit,
        settings.ASYNC_THROUGHPUT_PER_SEC,
        estimated_queue_delay_ms,
        max_queue_delay_ms,
    )

    if in_system > hard_limit or estimated_queue_delay_ms > max_queue_delay_ms:
        decrement_active()
        return False

    return True


def decrement_active() -> None:
    try:
        value = redis_client.decr(IN_SYSTEM_KEY)
        if value < 0:
            redis_client.set(IN_SYSTEM_KEY, "0")
    except Exception:
        # защита от падений Redis
        pass
