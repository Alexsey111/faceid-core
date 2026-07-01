"""Async verification job admission and enqueue helpers."""

import asyncio
import json
import logging
import time
import uuid
from time import perf_counter
from typing import Any, Dict, Optional, cast

import redis
from pydantic import BaseModel

from app.core.config import settings
from app.monitoring.metrics import (
    ASYNC_JOB_ENQUEUED_TOTAL,
    QUEUE_LENGTH_REDIS_SNAPSHOT,
    VERIFY_ACCEPTED_JOBS,
    VERIFY_REJECTED_JOBS,
    VERIFY_INFLIGHT_DECREMENT_TOTAL,
    VERIFY_INFLIGHT_INCREMENT_TOTAL,
    VERIFY_INFLIGHT_CURRENT,
    VERIFY_INFLIGHT_REDIS_SNAPSHOT,
    QUEUE_LENGTH,
    VERIFY_WORKER_UTILIZATION,
    REDIS_COMMAND_LATENCY_MS,
    QUEUE_PUSH_LATENCY_MS,
)

logger = logging.getLogger(__name__)

REDIS_POOL = redis.ConnectionPool(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    decode_responses=True,
    max_connections=50,
)

redis_client = redis.Redis(connection_pool=REDIS_POOL)

DECR_SCRIPT = """
local delta = tonumber(ARGV[1])
local val = redis.call("DECRBY", KEYS[1], delta)
if val < 0 then
    redis.call("SET", KEYS[1], 0)
    return 0
end
return val
"""


class QueueOverloadedError(RuntimeError):
    def __init__(
        self,
        reason: str,
        details: Dict[str, Any] | None = None,
        decision: Optional["AdmissionDecision"] = None,
    ):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}
        self.decision = decision


class AdmissionDecision(BaseModel):
    accepted: bool
    reason: str | None = None
    queue_len: int | None = None
    inflight: int | None = None
    estimated_delay_ms: float | None = None


class VerifyJobQueue:
    QUEUE_NAME = "face_verify_queue"

    @classmethod
    def inflight_limit(cls) -> int:
        return int(
            settings.INFLIGHT_LIMIT
            or max(10, settings.WORKER_COUNT * max(1, settings.WORKER_SEMAPHORE) * 2)
        )

    @classmethod
    def max_queue_size(cls) -> int:
        return int(
            settings.MAX_QUEUE_SIZE
            or max(8, settings.WORKER_COUNT * max(1, settings.WORKER_SEMAPHORE) * 2)
        )

    @classmethod
    def backpressure_max_queue_delay_ms(cls) -> float:
        return float(settings.BACKPRESSURE_MAX_QUEUE_DELAY_MS)

    @classmethod
    def async_throughput_per_sec(cls) -> float:
        return float(settings.ASYNC_THROUGHPUT_PER_SEC)

    @classmethod
    def admission_active_service_slots(cls) -> int:
        return int(
            settings.ADMISSION_ACTIVE_SERVICE_SLOTS
            or max(1, settings.WORKER_COUNT * max(1, settings.WORKER_SEMAPHORE))
        )

    @staticmethod
    def _get_inflight() -> int:
        start = perf_counter()
        inflight_raw = cast(str | None, redis_client.get("inflight_jobs"))
        REDIS_COMMAND_LATENCY_MS.labels(command="get_inflight_jobs").observe(
            (perf_counter() - start) * 1000.0
        )
        return int(inflight_raw) if inflight_raw else 0

    @staticmethod
    def _count_active_job_refs() -> tuple[int, int]:
        claim_count = 0
        lease_count = 0
        for _key in redis_client.scan_iter(match="job:*:claim", count=1000):
            claim_count += 1
        for _key in redis_client.scan_iter(match="job:*:lease", count=1000):
            lease_count += 1
        return claim_count, lease_count

    @classmethod
    async def get_queue_length(cls) -> int:
        return await asyncio.to_thread(cls._get_queue_length)

    @classmethod
    async def get_inflight_jobs(cls) -> int:
        return await asyncio.to_thread(cls._get_inflight)

    @classmethod
    async def reconcile_inflight_state(cls) -> dict[str, int | bool]:
        queue_len = await cls.get_queue_length()
        inflight = await cls.get_inflight_jobs()
        claim_count, lease_count = await asyncio.to_thread(cls._count_active_job_refs)

        reconciled = False
        if queue_len == 0 and claim_count == 0 and lease_count == 0:
            start = perf_counter()
            try:
                redis_client.set("inflight_jobs", 0)
                REDIS_COMMAND_LATENCY_MS.labels(command="set_inflight_jobs_zero").observe(
                    (perf_counter() - start) * 1000.0
                )
                inflight = 0
                reconciled = True
                cls._update_metrics(inflight, queue_len)
            except Exception:
                REDIS_COMMAND_LATENCY_MS.labels(command="set_inflight_jobs_zero").observe(
                    (perf_counter() - start) * 1000.0
                )
                raise
        else:
            cls._update_metrics(inflight, queue_len)

        return {
            "queue_len": queue_len,
            "inflight": inflight,
            "claim_count": claim_count,
            "lease_count": lease_count,
            "reconciled": reconciled,
        }

    @staticmethod
    def _update_metrics(inflight: int, queue_len: int | None = None) -> None:
        VERIFY_INFLIGHT_CURRENT.set(inflight)
        VERIFY_INFLIGHT_REDIS_SNAPSHOT.set(inflight)
        if queue_len is not None:
            QUEUE_LENGTH_REDIS_SNAPSHOT.set(float(queue_len))
        VERIFY_WORKER_UTILIZATION.set(
            min(1.0, inflight / float(VerifyJobQueue.inflight_limit()))
        )

    @staticmethod
    def record_inflight_increment(reason: str = "accepted") -> None:
        VERIFY_INFLIGHT_INCREMENT_TOTAL.labels(reason=reason).inc()

    @staticmethod
    def record_inflight_decrement(reason: str) -> None:
        VERIFY_INFLIGHT_DECREMENT_TOTAL.labels(reason=reason).inc()

    @staticmethod
    def finalize_job(job_id: str, terminal_state: str) -> int:
        start = perf_counter()
        try:
            inflight_raw = cast(
                int,
                redis_client.eval(DECR_SCRIPT, 1, "inflight_jobs", "1"),
            )
            REDIS_COMMAND_LATENCY_MS.labels(command="eval_decr_inflight_jobs").observe(
                (perf_counter() - start) * 1000.0
            )
            inflight = int(inflight_raw)
        except Exception:
            REDIS_COMMAND_LATENCY_MS.labels(command="eval_decr_inflight_jobs").observe(
                (perf_counter() - start) * 1000.0
            )
            raise

        VerifyJobQueue.record_inflight_decrement(terminal_state)
        VerifyJobQueue._update_metrics(inflight)
        try:
            redis_client.delete(f"job:{job_id}:claim", f"job:{job_id}:lease")
        except Exception:
            # Cleanup is best-effort; the inflight accounting has already been finalized.
            pass
        return inflight

    @staticmethod
    def _get_queue_length() -> int:
        start = perf_counter()
        queue_len = int(cast(int, redis_client.llen(VerifyJobQueue.QUEUE_NAME)))
        REDIS_COMMAND_LATENCY_MS.labels(command="llen_queue_pre_admission").observe(
            (perf_counter() - start) * 1000.0
        )
        QUEUE_LENGTH.set(float(queue_len))
        QUEUE_LENGTH_REDIS_SNAPSHOT.set(float(queue_len))
        return queue_len

    @staticmethod
    def _estimate_queue_delay_ms(inflight: int, queue_len: int) -> float:
        throughput = max(0.1, VerifyJobQueue.async_throughput_per_sec())
        effective_backlog = max(
            queue_len,
            max(0, inflight - VerifyJobQueue.admission_active_service_slots()),
        )
        return (effective_backlog / throughput) * 1000.0

    @classmethod
    def _evaluate_admission_from_values(
        cls,
        queue_len: int,
        inflight: int,
    ) -> AdmissionDecision:
        estimated_delay_ms = cls._estimate_queue_delay_ms(inflight=inflight, queue_len=queue_len)

        if inflight >= cls.inflight_limit():
            VERIFY_REJECTED_JOBS.labels(reason="inflight_limit").inc()
            return AdmissionDecision(
                accepted=False,
                reason="inflight_limit",
                queue_len=queue_len,
                inflight=inflight,
                estimated_delay_ms=estimated_delay_ms,
            )

        if queue_len >= cls.max_queue_size():
            VERIFY_REJECTED_JOBS.labels(reason="queue_overflow").inc()
            return AdmissionDecision(
                accepted=False,
                reason="queue_overflow",
                queue_len=queue_len,
                inflight=inflight,
                estimated_delay_ms=estimated_delay_ms,
            )

        if estimated_delay_ms >= cls.backpressure_max_queue_delay_ms():
            VERIFY_REJECTED_JOBS.labels(reason="estimated_delay").inc()
            return AdmissionDecision(
                accepted=False,
                reason="estimated_delay",
                queue_len=queue_len,
                inflight=inflight,
                estimated_delay_ms=estimated_delay_ms,
            )

        return AdmissionDecision(
            accepted=True,
            queue_len=queue_len,
            inflight=inflight,
            estimated_delay_ms=estimated_delay_ms,
        )

    @classmethod
    async def evaluate_admission(cls) -> AdmissionDecision:
        queue_len = await cls.get_queue_length()
        inflight = await cls.get_inflight_jobs()
        decision = cls._evaluate_admission_from_values(queue_len=queue_len, inflight=inflight)
        cls._update_metrics(inflight, queue_len)
        return decision

    @classmethod
    def enqueue(cls, payload: Dict[str, Any], admission: AdmissionDecision | None = None) -> str:
        decision = admission or cls._evaluate_admission_from_values(
            queue_len=cls._get_queue_length(),
            inflight=cls._get_inflight(),
        )

        if not decision.accepted:
            raise QueueOverloadedError(
                decision.reason or "admission_guard",
                {
                    "queue_length": decision.queue_len,
                    "inflight": decision.inflight,
                    "estimated_queue_delay_ms": round(decision.estimated_delay_ms or 0.0, 2),
                    "inflight_limit": cls.inflight_limit(),
                    "queue_limit": cls.max_queue_size(),
                    "max_queue_delay_ms": cls.backpressure_max_queue_delay_ms(),
                },
                decision=decision,
            )

        job_id = str(uuid.uuid4())

        job = {
            "job_id": job_id,
            "payload": payload,
            "created_at": time.time(),
        }

        inflight_incremented = False
        try:
            start = perf_counter()
            pipeline_factory: Any = getattr(redis_client, "pipeline", None)
            if callable(pipeline_factory):
                pipe: Any = pipeline_factory(transaction=True)
                pipe.rpush(VerifyJobQueue.QUEUE_NAME, json.dumps(job))
                pipe.setex(
                    f"job:{job_id}",
                    300,
                    json.dumps({"status": "processing"}),
                )
                pipe.llen(VerifyJobQueue.QUEUE_NAME)
                pipe.incr("inflight_jobs")
                push_start = perf_counter()
                _, _, queue_length_raw, incr_result = pipe.execute()
                QUEUE_PUSH_LATENCY_MS.observe((perf_counter() - push_start) * 1000.0)
                REDIS_COMMAND_LATENCY_MS.labels(command="enqueue_pipeline").observe(
                    (perf_counter() - start) * 1000.0
                )
                inflight_incremented = True

                inflight = int(cast(int, incr_result))
                VerifyJobQueue._update_metrics(inflight, int(cast(int, queue_length_raw)))

                VerifyJobQueue.record_inflight_increment("accepted")
                VERIFY_ACCEPTED_JOBS.inc()
                ASYNC_JOB_ENQUEUED_TOTAL.inc()
                QUEUE_LENGTH.set(float(cast(int, queue_length_raw)))

                return job_id

            push_start = perf_counter()
            redis_client.rpush(VerifyJobQueue.QUEUE_NAME, json.dumps(job))
            QUEUE_PUSH_LATENCY_MS.observe((perf_counter() - push_start) * 1000.0)
            redis_client.setex(
                f"job:{job_id}",
                300,
                json.dumps({"status": "processing"}),
            )
            queue_length_raw: Any = redis_client.llen(VerifyJobQueue.QUEUE_NAME)
            incr_result: Any = redis_client.incr("inflight_jobs")
            REDIS_COMMAND_LATENCY_MS.labels(command="enqueue_direct").observe(
                (perf_counter() - start) * 1000.0
            )
            inflight_incremented = True

            inflight = int(cast(int, incr_result))
            VerifyJobQueue._update_metrics(inflight, int(cast(int, queue_length_raw)))

            VerifyJobQueue.record_inflight_increment("accepted")
            VERIFY_ACCEPTED_JOBS.inc()
            ASYNC_JOB_ENQUEUED_TOTAL.inc()
            QUEUE_LENGTH.set(float(cast(int, queue_length_raw)))

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

    @classmethod
    async def enqueue_job(
        cls,
        payload: Dict[str, Any],
        admission: AdmissionDecision | None = None,
    ) -> str:
        return await asyncio.to_thread(cls.enqueue, payload, admission)
