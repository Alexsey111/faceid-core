from __future__ import annotations

from app.infrastructure.redis_client import redis_client

ACTIVE_KEY = "faceid:active_tasks"


def get_active_tasks() -> int:
    value = redis_client.get(ACTIVE_KEY)
    return int(value) if value else 0


def get_queue_length(queue_name: str = "faceid") -> int:
    try:
        return redis_client.llen(queue_name)
    except Exception:
        return 0


def increment_active() -> None:
    redis_client.incr(ACTIVE_KEY)


def decrement_active() -> None:
    try:
        value = redis_client.decr(ACTIVE_KEY)
        if value < 0:
            redis_client.set(ACTIVE_KEY, "0")
    except Exception:
        # защита от падений Redis
        pass
