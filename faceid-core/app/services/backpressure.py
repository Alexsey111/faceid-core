# faceid-core\app\services\backpressure.py
from __future__ import annotations

import logging

from app.core.config import settings
from app.infrastructure.redis_client import redis_client

IN_SYSTEM_KEY = "faceid:in_system"
logger = logging.getLogger(__name__)


def try_reserve_slot() -> bool:
    in_system = redis_client.incr(IN_SYSTEM_KEY)
    redis_client.expire(IN_SYSTEM_KEY, 60)
    logger.info(
        "[BP] try_reserve in_system=%s limit=%s",
        in_system,
        settings.MAX_ACTIVE_TASKS,
    )

    if in_system > settings.MAX_ACTIVE_TASKS:
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
