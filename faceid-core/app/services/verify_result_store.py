# app/services/verify_result_store.py

import json
import time
import redis
from typing import Any, Dict, Optional, cast

redis_client = redis.Redis(host="redis", port=6379, db=0)


class VerifyResultStore:

    TTL = 300

    @staticmethod
    def set_done(job_id: str, result: Dict[str, Any], metrics: Dict[str, Any]):
        completed_at = metrics.get("finished_at", time.time())
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

    @staticmethod
    def set_error(job_id: str, error: str, metrics: Dict[str, Any]):
        completed_at = metrics.get("finished_at", time.time())
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

    @staticmethod
    def set_expired(job_id: str, metrics: Dict[str, Any]):
        completed_at = metrics.get("finished_at", time.time())
        redis_client.setex(
            f"job:{job_id}",
            VerifyResultStore.TTL,
            json.dumps({
                "status": "expired",
                "metrics": metrics,
                "completed_at": completed_at,
            })
        )

    @staticmethod
    def get(job_id: str) -> Optional[Dict[str, Any]]:
        raw = redis_client.get(f"job:{job_id}")

        if raw is None:
            return None

        data = cast(bytes, raw)

        return json.loads(data.decode("utf-8"))
