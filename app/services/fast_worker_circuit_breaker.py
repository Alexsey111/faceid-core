# app\services\fast_worker_circuit_breaker.py

from __future__ import annotations

import threading

from app.core.config import settings


_lock = threading.Lock()
_enabled = bool(settings.FAST_WORKER_ENABLED)
_failures = max(0, int(settings.FAST_WORKER_FAILURES))


def is_fast_worker_enabled() -> bool:
    with _lock:
        return _enabled


def get_fast_worker_failures() -> int:
    with _lock:
        return _failures


def record_fast_worker_success() -> None:
    global _enabled, _failures
    with _lock:
        _enabled = True
        _failures = 0


def record_fast_worker_failure() -> int:
    global _enabled, _failures
    with _lock:
        _failures += 1
        if _failures > max(0, int(settings.FAST_WORKER_MAX_FAILURES)):
            _enabled = False
        return _failures
