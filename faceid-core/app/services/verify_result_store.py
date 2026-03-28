# app/services/verify_result_store.py

import json
import time
import redis
from time import perf_counter
from typing import Any, Dict, Optional, cast

from app.monitoring.metrics import REDIS_COMMAND_LATENCY_MS

REDIS_POOL = redis.ConnectionPool(
    host="redis",
    port=6379,
    db=0,
    decode_responses=True,
    max_connections=50,
)

redis_client = redis.Redis(connection_pool=REDIS_POOL)


class VerifyResultStore:

    TTL = 300

    @staticmethod
    def set_done(job_id: str, result: Dict[str, Any], metrics: Dict[str, Any]):
        completed_at = metrics.get("finished_at", time.time())
        start = perf_counter()
        redis_client.setex(
            f"job:{job_id}",
            VerifyResultStore.TTL,
            json.dumps({
                "status": "done",
                "result": result,
                "metrics": metrics,
                "completed_at": completed_at,
            })
        )
        REDIS_COMMAND_LATENCY_MS.labels(command="result_setex_done").observe(
            (perf_counter() - start) * 1000.0
        )

    @staticmethod
    def set_error(job_id: str, error: str, metrics: Dict[str, Any]):
        completed_at = metrics.get("finished_at", time.time())
        start = perf_counter()
        redis_client.setex(
            f"job:{job_id}",
            VerifyResultStore.TTL,
            json.dumps({
                "status": "error",
                "error": error,
                "metrics": metrics,
                "completed_at": completed_at,
            })
        )
        REDIS_COMMAND_LATENCY_MS.labels(command="result_setex_error").observe(
            (perf_counter() - start) * 1000.0
        )

    @staticmethod
    def set_expired(job_id: str, metrics: Dict[str, Any]):
        completed_at = metrics.get("finished_at", time.time())
        start = perf_counter()
        redis_client.setex(
            f"job:{job_id}",
            VerifyResultStore.TTL,
            json.dumps({
                "status": "expired",
                "metrics": metrics,
                "completed_at": completed_at,
            })
        )
        REDIS_COMMAND_LATENCY_MS.labels(command="result_setex_expired").observe(
            (perf_counter() - start) * 1000.0
        )

    @staticmethod
    def get(job_id: str) -> Optional[Dict[str, Any]]:
        start = perf_counter()
        raw = redis_client.get(f"job:{job_id}")
        REDIS_COMMAND_LATENCY_MS.labels(command="result_get").observe(
            (perf_counter() - start) * 1000.0
        )

        if raw is None:
            return None

        data = cast(str, raw)

        return json.loads(data)
