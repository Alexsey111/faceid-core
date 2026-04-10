# app/api/routes/job_status.py

import asyncio
import json

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.services.verify_result_store import VerifyResultStore

router = APIRouter()


def _extract_async_job_metrics(result: dict) -> dict[str, float | None]:
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return {
            "async_job_total_latency_ms": None,
            "async_job_queue_delay_ms": None,
            "async_job_processing_ms": None,
            "job_created_at": None,
            "job_started_at": None,
            "job_done_at": result.get("completed_at"),
        }

    def to_ms(value: object) -> float | None:
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
        "async_job_total_latency_ms": to_ms(metrics.get("total_latency")),
        "async_job_queue_delay_ms": to_ms(metrics.get("queue_delay")),
        "async_job_processing_ms": to_ms(metrics.get("processing_time")),
        "job_created_at": metrics.get("created_at"),
        "job_started_at": metrics.get("started_at"),
        "job_done_at": result.get("completed_at") or metrics.get("finished_at"),
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
    deadline = asyncio.get_running_loop().time() + (timeout / 1000.0)

    while True:
        result = VerifyResultStore.get(job_id)
        if result:
            status = result.get("status")
            if status in {"done", "error", "expired", "failed"}:
                return {
                    "job_id": job_id,
                    **_extract_async_job_metrics(result),
                    **result,
                }

        if asyncio.get_running_loop().time() >= deadline:
            return {
                "job_id": job_id,
                "status": "processing",
            }

        await asyncio.sleep(0.05)
