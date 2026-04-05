# app/services/verify_job_queue.py

import json
import uuid
import time
import redis
from typing import Any, Dict, cast
from time import perf_counter

from app.core.config import settings
from app.monitoring.metrics import (
    VERIFY_ACCEPTED_JOBS,
    VERIFY_INFLIGHT_CURRENT,
    QUEUE_LENGTH,
    VERIFY_WORKER_UTILIZATION,
    REDIS_COMMAND_LATENCY_MS,
    QUEUE_PUSH_LATENCY_MS,
)

REDIS_POOL = redis.ConnectionPool(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    decode_responses=True,
    max_connections=50,
)

redis_client = redis.Redis(connection_pool=REDIS_POOL)


class VerifyJobQueue:
    QUEUE_NAME = "face_verify_queue"
    INFLIGHT_LIMIT = 30
    MAX_QUEUE_SIZE = 250

    @staticmethod
    def _get_inflight() -> int:
        start = perf_counter()
        inflight_raw = cast(str | None, redis_client.get("inflight_jobs"))
        REDIS_COMMAND_LATENCY_MS.labels(command="get_inflight_jobs").observe(
            (perf_counter() - start) * 1000.0
        )
        return int(inflight_raw) if inflight_raw else 0

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

        job_id = str(uuid.uuid4())

        job = {
            "job_id": job_id,
            "payload": payload,
            "created_at": time.time(),
        }

        inflight_incremented = False
        try:
            start = perf_counter()
            redis_client.incr("inflight_jobs")
            REDIS_COMMAND_LATENCY_MS.labels(command="incr_inflight_jobs").observe(
                (perf_counter() - start) * 1000.0
            )
            inflight_incremented = True

            inflight = VerifyJobQueue._get_inflight()
            VerifyJobQueue._update_metrics(inflight)

            start = perf_counter()
            redis_client.rpush(VerifyJobQueue.QUEUE_NAME, json.dumps(job))
            QUEUE_PUSH_LATENCY_MS.observe((perf_counter() - start) * 1000.0)
            VERIFY_ACCEPTED_JOBS.inc()
            start = perf_counter()
            queue_length = float(cast(int, redis_client.llen(VerifyJobQueue.QUEUE_NAME)))
            REDIS_COMMAND_LATENCY_MS.labels(command="llen_queue_post_push").observe(
                (perf_counter() - start) * 1000.0
            )
            QUEUE_LENGTH.set(queue_length)

            # mark as processing immediately
            redis_client.setex(
                f"job:{job_id}",
                300,
                json.dumps({"status": "processing"}),
            )

            return job_id
        except Exception:
            if inflight_incremented:
                try:
                    start = perf_counter()
                    redis_client.decr("inflight_jobs")
                    REDIS_COMMAND_LATENCY_MS.labels(command="decr_inflight_jobs").observe(
                        (perf_counter() - start) * 1000.0
                    )
                finally:
                    inflight = VerifyJobQueue._get_inflight()
                    VerifyJobQueue._update_metrics(inflight)
            raise
