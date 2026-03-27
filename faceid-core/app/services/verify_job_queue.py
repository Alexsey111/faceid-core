# app/services/verify_job_queue.py

import json
import uuid
import time
import redis
from typing import Any, Dict, cast

from app.core.config import settings
from app.monitoring.metrics import (
    VERIFY_ACCEPTED_JOBS,
    VERIFY_INFLIGHT_CURRENT,
    QUEUE_LENGTH,
    VERIFY_WORKER_UTILIZATION,
)

redis_client = redis.Redis(host="redis", port=6379, db=0)


class VerifyJobQueue:
    QUEUE_NAME = "face_verify_queue"
    INFLIGHT_LIMIT = 10
    MAX_QUEUE_SIZE = 100

    @staticmethod
    def _get_inflight() -> int:
        inflight_raw = cast(bytes | None, redis_client.get("inflight_jobs"))
        return int(inflight_raw.decode("utf-8")) if inflight_raw else 0

    @staticmethod
    def _update_metrics(inflight: int) -> None:
        VERIFY_INFLIGHT_CURRENT.set(inflight)
        VERIFY_WORKER_UTILIZATION.set(
            min(1.0, inflight / float(VerifyJobQueue.INFLIGHT_LIMIT))
        )

    @staticmethod
    def enqueue(payload: Dict[str, Any]) -> str:
        inflight = VerifyJobQueue._get_inflight()
        VerifyJobQueue._update_metrics(inflight)

        if inflight >= VerifyJobQueue.INFLIGHT_LIMIT * 3:
            raise Exception("overloaded")

        queue_size = cast(int, redis_client.llen(VerifyJobQueue.QUEUE_NAME))
        throughput_per_sec = max(0.001, float(settings.ASYNC_THROUGHPUT_PER_SEC))
        estimated_delay_ms = (queue_size / throughput_per_sec) * 1000.0

        if estimated_delay_ms > float(settings.BACKPRESSURE_MAX_QUEUE_DELAY_MS):
            raise Exception("sla_overflow")

        if queue_size >= VerifyJobQueue.MAX_QUEUE_SIZE:
            # не сразу reject — даём шанс
            if queue_size >= VerifyJobQueue.MAX_QUEUE_SIZE * 1.5:
                raise Exception("queue_overflow")

        job_id = str(uuid.uuid4())

        job = {
            "job_id": job_id,
            "payload": payload,
            "created_at": time.time(),
        }

        redis_client.rpush(VerifyJobQueue.QUEUE_NAME, json.dumps(job))
        redis_client.incr("inflight_jobs")
        VERIFY_ACCEPTED_JOBS.inc()
        queue_length = float(cast(int, redis_client.llen(VerifyJobQueue.QUEUE_NAME)))
        QUEUE_LENGTH.set(queue_length)

        inflight = VerifyJobQueue._get_inflight()
        VerifyJobQueue._update_metrics(inflight)

        # mark as processing immediately
        redis_client.setex(
            f"job:{job_id}",
            300,
            json.dumps({"status": "processing"}),
        )

        return job_id
