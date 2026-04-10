# faceid-core\app\monitoring\db_metrics.py

from __future__ import annotations

from time import perf_counter
from typing import Awaitable, TypeVar

try:
    from app.monitoring.metrics import DB_QUERY_TIME_MS
    METRICS_ENABLED = True
except Exception:
    METRICS_ENABLED = False
    DB_QUERY_TIME_MS = None  # type: ignore[assignment]

T = TypeVar("T")


async def timed_db_call(awaitable: Awaitable[T], operation: str) -> T:
    start = perf_counter()
    try:
        return await awaitable
    finally:
        if METRICS_ENABLED and DB_QUERY_TIME_MS is not None:
            try:
                DB_QUERY_TIME_MS.labels(operation=operation).observe(
                    (perf_counter() - start) * 1000.0
                )
            except Exception:
                pass
