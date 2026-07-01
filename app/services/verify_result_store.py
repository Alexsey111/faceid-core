# app/services/verify_result_store.py

import json
import redis
from time import perf_counter
from typing import Any, Dict, Optional, cast

from app.core.config import settings
from app.monitoring.metrics import (
    ASYNC_JOB_TERMINAL_TOTAL,
    REDIS_COMMAND_LATENCY_MS,
    VERIFY_RESULT_VISIBLE_LAG_MS,
    VERIFY_RESULT_WRITE_MS,
    VERIFY_RESULT_VISIBLE_TOTAL,
)

REDIS_POOL = redis.ConnectionPool(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    decode_responses=True,
    max_connections=50,
)

redis_client = redis.Redis(connection_pool=REDIS_POOL)


class VerifyResultStore:

    TTL = 300

    _SENSITIVE_KEYS = {
        "completed_at",
        "error",
        "metrics",
        "image",
        "image_b64",
        "image_bytes",
        "raw_image",
        "face_input",
        "embedding",
        "query_embedding",
        "vector",
        "payload",
        "full_payload",
        "bytes",
        "service_timings",
        "timings",
        "timestamps",
        "job_timings",
    }

    @staticmethod
    def _sanitize_value(value: Any) -> Any:
        if isinstance(value, dict):
            return VerifyResultStore._sanitize_mapping(value)
        if isinstance(value, list):
            return [VerifyResultStore._sanitize_value(item) for item in value]
        if isinstance(value, tuple):
            return [VerifyResultStore._sanitize_value(item) for item in value]
        if isinstance(value, (bytes, bytearray, memoryview)):
            return None
        return value

    @staticmethod
    def _sanitize_mapping(data: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for key, value in data.items():
            if key in VerifyResultStore._SENSITIVE_KEYS:
                continue
            sanitized[key] = VerifyResultStore._sanitize_value(value)
        return sanitized

    @staticmethod
    def _extract_timings(source: Dict[str, Any] | None) -> Dict[str, Any]:
        if not isinstance(source, dict):
            return {}
        timings: dict[str, Any] = {}
        raw_timings = source.get("timings")
        if isinstance(raw_timings, dict):
            timings.update(VerifyResultStore._sanitize_mapping(cast(Dict[str, Any], raw_timings)))
        raw_service_timings = source.get("service_timings")
        if isinstance(raw_service_timings, dict):
            timings.update(VerifyResultStore._sanitize_mapping(cast(Dict[str, Any], raw_service_timings)))
        return timings

    @staticmethod
    def _extract_timestamps(source: Dict[str, Any] | None, metrics: Dict[str, Any]) -> Dict[str, Any]:
        timestamps: dict[str, Any] = {}
        if isinstance(source, dict):
            raw_timestamps = source.get("timestamps")
            if isinstance(raw_timestamps, dict):
                timestamps.update(VerifyResultStore._sanitize_mapping(cast(Dict[str, Any], raw_timestamps)))
        return timestamps

    @staticmethod
    def _extract_technical_timestamps(
        source: Dict[str, Any] | None,
        technical_timestamps: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        technical: dict[str, Any] = {}
        if isinstance(source, dict):
            for key in (
                "completed_at_ms",
                "result_visible_at_ms",
                "worker_started_at_ms",
                "queue_popped_at_ms",
            ):
                value = source.get(key)
                if value is not None:
                    technical[key] = value
        if isinstance(technical_timestamps, dict):
            for key, value in technical_timestamps.items():
                if value is not None:
                    technical[key] = value
        return technical

    @staticmethod
    def _build_envelope(
        *,
        status: str,
        result: Dict[str, Any] | None = None,
        metrics: Dict[str, Any] | None = None,
        technical_timestamps: Dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = result or {}
        metrics = metrics or {}
        technical = VerifyResultStore._extract_technical_timestamps(result, technical_timestamps)
        envelope: dict[str, Any] = {
            "status": status,
            "result": VerifyResultStore._sanitize_mapping(result),
            "timings": VerifyResultStore._extract_timings(result),
            "timestamps": VerifyResultStore._extract_timestamps(result, metrics),
        }
        envelope.update(technical)
        return envelope

    @staticmethod
    def set_done(
        job_id: str,
        result: Dict[str, Any],
        metrics: Dict[str, Any] | None = None,
        technical_timestamps: Dict[str, Any] | None = None,
    ):
        start = perf_counter()
        ASYNC_JOB_TERMINAL_TOTAL.labels(status="done").inc()
        payload = VerifyResultStore._build_envelope(
            status="done",
            result=result,
            metrics=metrics,
            technical_timestamps=technical_timestamps,
        )
        redis_client.setex(
            f"job:{job_id}",
            VerifyResultStore.TTL,
            json.dumps(payload)
        )
        REDIS_COMMAND_LATENCY_MS.labels(command="result_setex_done").observe(
            (perf_counter() - start) * 1000.0
        )
        VERIFY_RESULT_WRITE_MS.observe((perf_counter() - start) * 1000.0)
        VERIFY_RESULT_VISIBLE_TOTAL.labels(status="done").inc()
        completed_at = technical_timestamps.get("completed_at_ms") if isinstance(technical_timestamps, dict) else None
        result_visible_at = technical_timestamps.get("result_visible_at_ms") if isinstance(technical_timestamps, dict) else None
        if completed_at is not None and result_visible_at is not None:
            try:
                VERIFY_RESULT_VISIBLE_LAG_MS.observe(max(0.0, float(result_visible_at) - float(completed_at)))
            except Exception:
                pass

    @staticmethod
    def set_error(
        job_id: str,
        error: str,
        metrics: Dict[str, Any] | None = None,
        technical_timestamps: Dict[str, Any] | None = None,
    ):
        start = perf_counter()
        ASYNC_JOB_TERMINAL_TOTAL.labels(status="error").inc()
        payload = VerifyResultStore._build_envelope(
            status="error",
            result={"error": error},
            metrics=metrics,
            technical_timestamps=technical_timestamps,
        )
        redis_client.setex(
            f"job:{job_id}",
            VerifyResultStore.TTL,
            json.dumps(payload)
        )
        REDIS_COMMAND_LATENCY_MS.labels(command="result_setex_error").observe(
            (perf_counter() - start) * 1000.0
        )
        VERIFY_RESULT_WRITE_MS.observe((perf_counter() - start) * 1000.0)
        VERIFY_RESULT_VISIBLE_TOTAL.labels(status="error").inc()
        completed_at = technical_timestamps.get("completed_at_ms") if isinstance(technical_timestamps, dict) else None
        result_visible_at = technical_timestamps.get("result_visible_at_ms") if isinstance(technical_timestamps, dict) else None
        if completed_at is not None and result_visible_at is not None:
            try:
                VERIFY_RESULT_VISIBLE_LAG_MS.observe(max(0.0, float(result_visible_at) - float(completed_at)))
            except Exception:
                pass

    @staticmethod
    def set_expired(
        job_id: str,
        metrics: Dict[str, Any] | None = None,
        technical_timestamps: Dict[str, Any] | None = None,
    ):
        start = perf_counter()
        ASYNC_JOB_TERMINAL_TOTAL.labels(status="expired").inc()
        payload = VerifyResultStore._build_envelope(
            status="expired",
            result={},
            metrics=metrics,
            technical_timestamps=technical_timestamps,
        )
        redis_client.setex(
            f"job:{job_id}",
            VerifyResultStore.TTL,
            json.dumps(payload)
        )
        REDIS_COMMAND_LATENCY_MS.labels(command="result_setex_expired").observe(
            (perf_counter() - start) * 1000.0
        )
        VERIFY_RESULT_WRITE_MS.observe((perf_counter() - start) * 1000.0)
        VERIFY_RESULT_VISIBLE_TOTAL.labels(status="expired").inc()
        completed_at = technical_timestamps.get("completed_at_ms") if isinstance(technical_timestamps, dict) else None
        result_visible_at = technical_timestamps.get("result_visible_at_ms") if isinstance(technical_timestamps, dict) else None
        if completed_at is not None and result_visible_at is not None:
            try:
                VERIFY_RESULT_VISIBLE_LAG_MS.observe(max(0.0, float(result_visible_at) - float(completed_at)))
            except Exception:
                pass

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
