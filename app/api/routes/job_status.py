# app/api/routes/job_status.py

import asyncio
import json
import time

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.monitoring.metrics import (
    VERIFY_TERMINAL_GAP_MS,
    VERIFY_VISIBLE_TO_FIRST_HIT_MS,
    VERIFY_WAIT_EMPTY_CYCLES,
    VERIFY_WAIT_HIT_TOTAL,
    VERIFY_WAIT_LOOKUP_MS,
    VERIFY_WAIT_MISS_TOTAL,
    VERIFY_WAIT_REQUEST_TOTAL,
    VERIFY_WAIT_TIMEOUT_TOTAL,
    VERIFY_WAIT_HOLD_MS,
)
from app.services.verify_result_store import VerifyResultStore

router = APIRouter()


def _to_float_ms(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_result_technical_ms(result: dict) -> dict[str, float | None]:
    timestamps = result.get("timestamps") if isinstance(result.get("timestamps"), dict) else {}
    keys = (
        "completed_at_ms",
        "result_visible_at_ms",
        "worker_started_at_ms",
        "queue_popped_at_ms",
    )
    technical: dict[str, float | None] = {}
    for key in keys:
        value = result.get(key)
        if value is None and isinstance(timestamps, dict):
            value = timestamps.get(key)
        technical[key] = _to_float_ms(value)
    return technical


def _extract_async_job_metrics(result: dict) -> dict[str, float | None]:
    timings = result.get("timings")
    timestamps = result.get("timestamps")

    def ns_to_s(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value) / 1_000_000_000.0
        if isinstance(value, str):
            try:
                return float(value) / 1_000_000_000.0
            except ValueError:
                return None
        return None

    def to_ms(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    if isinstance(timings, dict) or isinstance(timestamps, dict):
        queue_delay_ms = to_ms(timings.get("queue_wait_ms")) if isinstance(timings, dict) else None
        processing_ms = to_ms(timings.get("pipeline_total_ms")) if isinstance(timings, dict) else None
        if processing_ms is None and isinstance(timings, dict):
            processing_ms = to_ms(timings.get("result_write_ms"))
        total_latency_ms = to_ms(timings.get("job_total_server_ms")) if isinstance(timings, dict) else None
        if total_latency_ms is None and isinstance(timestamps, dict):
            accepted_at = ns_to_s(timestamps.get("accepted_at_ns"))
            result_written_at = ns_to_s(timestamps.get("result_written_at_ns"))
            if accepted_at is not None and result_written_at is not None:
                total_latency_ms = (result_written_at - accepted_at) * 1000.0
        if processing_ms is None and isinstance(timestamps, dict):
            processing_started_at = ns_to_s(timestamps.get("processing_started_at_ns"))
            processing_finished_at = ns_to_s(timestamps.get("processing_finished_at_ns"))
            if processing_started_at is not None and processing_finished_at is not None:
                processing_ms = (processing_finished_at - processing_started_at) * 1000.0

        return {
            "async_job_total_latency_ms": total_latency_ms,
            "async_job_queue_delay_ms": queue_delay_ms,
            "async_job_processing_ms": processing_ms,
            "job_created_at": ns_to_s(timestamps.get("accepted_at_ns")) if isinstance(timestamps, dict) else None,
            "job_started_at": ns_to_s(timestamps.get("processing_started_at_ns")) if isinstance(timestamps, dict) else None,
            "job_done_at": ns_to_s(timestamps.get("result_written_at_ns")) if isinstance(timestamps, dict) else None,
        }

    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return {
            "async_job_total_latency_ms": None,
            "async_job_queue_delay_ms": None,
            "async_job_processing_ms": None,
            "job_created_at": None,
            "job_started_at": None,
            "job_done_at": None,
        }

    def metrics_to_ms(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value) * 1000.0
        if isinstance(value, str):
            try:
                return float(value) * 1000.0
            except ValueError:
                return None
        return None

    return {
        "async_job_total_latency_ms": metrics_to_ms(metrics.get("total_latency")),
        "async_job_queue_delay_ms": metrics_to_ms(metrics.get("queue_delay")),
        "async_job_processing_ms": metrics_to_ms(metrics.get("processing_time")),
        "job_created_at": metrics.get("created_at"),
        "job_started_at": metrics.get("started_at"),
        "job_done_at": metrics.get("finished_at"),
    }


async def get_job_from_redis(job_id: str) -> dict | None:
    return VerifyResultStore.get(job_id)


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    result = VerifyResultStore.get(job_id)

    if not result:
        return {
            "job_id": job_id,
            "status": "not_found",
        }

    return {"job_id": job_id, **result}


@router.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    async def event_stream():
        while True:
            job = await get_job_from_redis(job_id)

            if job and job.get("status") == "done":
                yield f"data: {json.dumps({'job_id': job_id, **job})}\n\n"
                break

            await asyncio.sleep(0.05)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/jobs/{job_id}/wait")
async def wait_job(job_id: str, timeout: int = Query(2000, ge=0, le=30000)):
    VERIFY_WAIT_REQUEST_TOTAL.inc()
    route_start_ns = time.perf_counter_ns()
    wait_block_start_ns = time.perf_counter_ns()
    lookup_total_ns = 0
    empty_cycles = 0
    first_result_hit = False
    result_visible_age_ms: float | None = None
    poll_cycles = 0
    deadline = asyncio.get_running_loop().time() + (timeout / 1000.0)

    while True:
        fetch_start_ns = time.perf_counter_ns()
        result = VerifyResultStore.get(job_id)
        fetch_elapsed_ns = time.perf_counter_ns() - fetch_start_ns
        lookup_total_ns += fetch_elapsed_ns
        result_fetch_ms = fetch_elapsed_ns / 1_000_000.0
        poll_cycles += 1
        if result:
            status = result.get("status")
            if status in {"done", "error", "expired", "failed"}:
                VERIFY_WAIT_HIT_TOTAL.inc()
                technical_ms = _extract_result_technical_ms(result)
                result_visible_at_ms = technical_ms.get("result_visible_at_ms")
                if result_visible_at_ms is not None:
                    hit_at_ms = time.time() * 1000.0
                    result_visible_age_ms = max(0.0, hit_at_ms - result_visible_at_ms)
                    VERIFY_VISIBLE_TO_FIRST_HIT_MS.observe(result_visible_age_ms)
                first_result_hit = empty_cycles == 0
                wait_block_ms = (time.perf_counter_ns() - wait_block_start_ns) / 1_000_000.0
                wait_route_ms = (time.perf_counter_ns() - route_start_ns) / 1_000_000.0
                VERIFY_WAIT_HOLD_MS.observe(wait_block_ms)
                VERIFY_WAIT_LOOKUP_MS.observe(lookup_total_ns / 1_000_000.0)
                if empty_cycles > 0:
                    VERIFY_WAIT_EMPTY_CYCLES.inc(empty_cycles)
                if result_visible_age_ms is not None:
                    VERIFY_TERMINAL_GAP_MS.observe(max(0.0, wait_block_ms - result_visible_age_ms))
                return {
                    "job_id": job_id,
                    **_extract_async_job_metrics(result),
                    "server_wait_ms": wait_block_ms,
                    "wait_lookup_ms": lookup_total_ns / 1_000_000.0,
                    "result_visible_age_ms": result_visible_age_ms,
                    "wait_empty_cycles": empty_cycles,
                    "poll_cycles": poll_cycles,
                    "first_result_hit": first_result_hit,
                    "wait_route_ms": wait_route_ms,
                    "wait_block_ms": wait_block_ms,
                    "result_fetch_ms": result_fetch_ms,
                    **result,
                }

        VERIFY_WAIT_MISS_TOTAL.inc()
        empty_cycles += 1
        if asyncio.get_running_loop().time() >= deadline:
            VERIFY_WAIT_TIMEOUT_TOTAL.inc()
            wait_block_ms = (time.perf_counter_ns() - wait_block_start_ns) / 1_000_000.0
            wait_route_ms = (time.perf_counter_ns() - route_start_ns) / 1_000_000.0
            VERIFY_WAIT_HOLD_MS.observe(wait_block_ms)
            VERIFY_WAIT_LOOKUP_MS.observe(lookup_total_ns / 1_000_000.0)
            if empty_cycles > 0:
                VERIFY_WAIT_EMPTY_CYCLES.inc(empty_cycles)
            return {
                "job_id": job_id,
                "status": "processing",
                "server_wait_ms": wait_block_ms,
                "wait_lookup_ms": lookup_total_ns / 1_000_000.0,
                "result_visible_age_ms": result_visible_age_ms,
                "wait_empty_cycles": empty_cycles,
                "poll_cycles": poll_cycles,
                "first_result_hit": first_result_hit,
                "wait_route_ms": wait_route_ms,
                "wait_block_ms": wait_block_ms,
                "result_fetch_ms": result_fetch_ms,
            }

        await asyncio.sleep(0.05)
