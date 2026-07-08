# app/api/routes/verify.py

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.verification_job_repo import VerificationJobRepository
from app.core.config import settings
from app.core.timing import now_epoch_ns
from app.api._helpers import MAX_IMAGE_SIZE, get_request_id
from app.db.session import get_db
from app.infrastructure.minio_client import MinioClient
from app.infrastructure.redis_client import redis_client
from app.models.verification_job import JobStatus
from app.schemas.verify import VerifyEnqueueResponse, VerifyRequest, VerifyResponse
from app.services.backpressure import (
    decrement_active,
    current_active_requests,
    get_backpressure_mode,
    get_system_load,
    should_use_async,
    should_drop_request,
    try_reserve_fast_path_slot,
    try_reserve_slot,
)
from app.services.fast_worker_circuit_breaker import (
    get_fast_worker_failures,
    is_fast_worker_enabled,
    record_fast_worker_failure,
    record_fast_worker_success,
)
from app.services.rate_limiter import RateLimiter
from app.services.rate_limiter import get_inflight_limit, get_queue_delay
from app.services.verification_service_factory import (
    get_verification_service,
    get_verification_service_without_pipeline,
)
from app.services.verify_job_queue import VerifyJobQueue
from app.services.webhook_service import fire_sync_webhook

router = APIRouter()
logger = logging.getLogger(__name__)

_fast_worker_client: httpx.AsyncClient | None = None


def _normalize_priority(value: str | None) -> tuple[str, int]:
    """Валидация priority ({high, low}) → 400 при некорректном.

    Celery-приоритет (возвращаемый int) больше не используется — постановка идёт в
    единую face_verify_queue (VerifyJobQueue), приоритизация — на стороне worker'а.
    Валидация сохранена для обратно-совместимого 400-контракта legacy-роутов
    (/verify_async_file, /verify_async_base64, /verify_base64 fallback).
    """
    priority = (value or "high").strip().lower()
    if priority not in {"high", "low"}:
        raise HTTPException(status_code=400, detail="Invalid priority")

    return priority, 9 if priority == "high" else 0


def _resolve_liveness(request: VerifyRequest) -> tuple[bool, bool]:
    """Решить режим liveness для /verify.

    Returns:
        (effective_require_liveness, active_proven).

    При liveness_mode="active": валидирует + consumes liveness_token (single-use,
    403 при невалидном/просроченном), effective_require_liveness=False (passive
    не запускается повторно — liveness уже доказан challenge-протоколом),
    active_proven=True → ответ получит liveness_passed=True.
    Иначе — request.require_liveness, active_proven=False.
    """
    mode = (request.liveness_mode or "passive").strip().lower()
    if mode == "active":
        from app.services.liveness_token import consume_liveness_token

        if not consume_liveness_token(request.liveness_token):
            raise HTTPException(
                status_code=403, detail="invalid or expired liveness_token"
            )
        logger.info("active_liveness_verified token consumed")
        return False, True
    # Active-challenge gate допуска: при require_liveness=true и
    # LIVENESS_ACTIVE_REQUIRED=true — passive-запрос на допуск отвергается
    # (physical-spoof cutout/print ложит passive-модель; active challenge их
    # отбрасывает). Направляем клиента пройти /liveness/challenge и повторить с
    # liveness_mode=active. require_liveness=false гейт не затрагивает.
    if request.require_liveness and settings.LIVENESS_ACTIVE_REQUIRED:
        raise HTTPException(
            status_code=403,
            detail=(
                "active_liveness_required: call /api/v1/liveness/challenge/init then "
                "WS /api/v1/liveness/challenge/stream to obtain liveness_token, "
                "retry /verify with liveness_mode=active"
            ),
        )
    return request.require_liveness, False


def _apply_active_liveness(result: Any, active_proven: bool) -> Any:
    """При active-proven выставить liveness_passed=True в ответе (dict или модель)."""
    if not active_proven:
        return result
    if isinstance(result, dict):
        result["liveness_passed"] = True
    elif hasattr(result, "liveness_passed"):
        try:
            result.liveness_passed = True
        except Exception:
            pass
    return result


def _pick_fast_worker_url() -> str:
    return settings.FAST_WORKER_URL


def get_fast_worker_client() -> httpx.AsyncClient:
    global _fast_worker_client

    if _fast_worker_client is None:
        _fast_worker_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=1.0,
                read=2.0,
                write=1.0,
                pool=1.0,
            ),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=50,
            ),
        )

    return _fast_worker_client


async def _call_fast_worker(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, object],
) -> tuple[dict[str, object], float]:
    t0 = time.perf_counter()
    resp = await client.post(f"{url}/verify_sync", json=payload)
    upstream_http_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return data, upstream_http_ms

    terminal_statuses = {
        "no_face",
        "spoof",
        "spoof_detected",
        "quality_reject",
        "retry",
        "processing_failed",
    }

    if data.get("status") in terminal_statuses:
        return data, upstream_http_ms

    if "embedding" not in data or data["embedding"] is None:
        raise HTTPException(status_code=502, detail="fast_worker returned invalid payload")
    return data, upstream_http_ms


async def _enqueue_verify_job(
    *,
    db: AsyncSession,
    job_id: str,
    request_received_time: float,
    image_bytes: bytes,
    object_name: str,
    content_type: str,
    user_id: str | None,
    require_liveness: bool,
    priority: str,
) -> None:
    job_repo = VerificationJobRepository(db)
    await job_repo.create(job_id=job_id, status=JobStatus.pending)
    await db.commit()

    # Валидация priority (400 при некорректном) — до тяжёлой MinIO/redis-работы.
    priority_name, _ = _normalize_priority(priority)
    minio_client = MinioClient()

    try:
        await asyncio.to_thread(
            minio_client.upload_image,
            object_name,
            image_bytes,
            content_type,
        )
        # Постановка в face_verify_queue (потребляется app.workers.verify_worker).
        # Раньше использовался Celery (verify_heavy/verify_fast), но в default-deploy
        # Celery-worker'ы отключены (profiles: disabled) → задания зависали pending.
        # VerifyJobQueue использует тот же worker, что и новые async-роуты, и полностью
        # обрабатывает no-face/quality_reject/spoof (ловит pipeline ValueError →
        # terminal). job_id передаём свой — он уже зафиксирован в DB/MinIO и отдан
        # клиенту для поллинга /jobs/{id}/wait.
        accepted_at_ns = int(request_received_time * 1_000_000_000)
        enqueue_payload = {
            "image_url": object_name,
            "user_id": user_id,
            "require_liveness": require_liveness,
            "accepted_at_ns": accepted_at_ns,
            "enqueued_at_ns": now_epoch_ns(),
        }
        await VerifyJobQueue.enqueue_job(enqueue_payload, admission=None, job_id=job_id)
    except Exception as exc:
        await job_repo.update(job_id, status=JobStatus.failed, error=str(exc))
        await db.commit()
        raise HTTPException(status_code=500, detail="Failed to enqueue verify job")

    logger.info(
        "verify_async_enqueued",
        extra={
            "job_id": job_id,
            "image_url": object_name,
            "queue": VerifyJobQueue.QUEUE_NAME,
            "priority": priority,
        },
    )

    return None


@router.post("/verify", response_model=VerifyResponse)
async def verify_file(
    http_request: Request,
    file: UploadFile = File(...),
    user_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Verify face against reference via multipart/form-data."""
    queue_delay = get_queue_delay()
    dynamic_limit = get_inflight_limit(queue_delay)

    RateLimiter.check(http_request, "verify", limit=dynamic_limit)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image format")

    image_bytes = await file.read()

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="Image too large")

    service = get_verification_service(db)

    result = await service.verify_face(
        image_bytes,
        user_id=user_id,
    )

    fire_sync_webhook(result)

    return result


@router.post("/verify_base64", response_model=VerifyResponse | VerifyEnqueueResponse)
async def verify_base64(
    request: VerifyRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Fast-path sync verify for low-load use, with Celery fallback."""
    queue_delay = get_queue_delay()
    dynamic_limit = get_inflight_limit(queue_delay)

    RateLimiter.check(http_request, "verify", limit=dynamic_limit)

    # Active liveness: валидируем+consumes token ДО тяжёлой работы (single-use).
    effective_require_liveness, active_proven = _resolve_liveness(request)

    image_bytes = base64.b64decode(request.image)

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="Image too large")

    if settings.USE_FAST_PATH and is_fast_worker_enabled():
        if not should_use_async():
            if try_reserve_fast_path_slot():
                try:
                    t_start = time.time()
                    client = get_fast_worker_client()
                    pipeline_result, upstream_http_ms = await _call_fast_worker(
                        client,
                        _pick_fast_worker_url(),
                        request.model_dump(),
                    )
                    logger.warning(
                        "fast_worker_timing upstream_http_ms=%.2f worker_total_ms=%s wait_for_slot_ms=%s",
                        upstream_http_ms,
                        pipeline_result.get("worker_total_ms"),
                        pipeline_result.get("wait_for_slot_ms"),
                    )
                    logger.warning(
                        "fast_worker_identity worker_hostname=%s worker_pid=%s",
                        pipeline_result.get("worker_hostname"),
                        pipeline_result.get("worker_pid"),
                    )
                    record_fast_worker_success()

                    service = get_verification_service_without_pipeline(db)
                    result = await service.verify_from_pipeline_result(
                        pipeline_result,
                        image_bytes=image_bytes,
                        user_id=request.user_id,
                        require_liveness=effective_require_liveness,
                        check_replay=True,
                        t_start=t_start,
                    )
                    result = _apply_active_liveness(result, active_proven)
                    fire_sync_webhook(result)
                    return result
                except Exception as exc:
                    failures = record_fast_worker_failure()
                    logger.warning(
                        "fast_worker_unavailable, falling back to async queue failures=%s enabled=%s error=%s",
                        failures,
                        is_fast_worker_enabled(),
                        exc,
                    )
                finally:
                    decrement_active()
    elif settings.USE_FAST_PATH:
        logger.warning(
            "fast_worker_circuit_open, using celery fallback failures=%s",
            get_fast_worker_failures(),
        )

    job_id = get_request_id(http_request)
    request_received_time = time.time()
    safe_filename = "legacy.jpg"
    object_name = f"verify/{job_id}/{safe_filename}"

    await _enqueue_verify_job(
        db=db,
        job_id=job_id,
        request_received_time=request_received_time,
        image_bytes=image_bytes,
        object_name=object_name,
        content_type="image/jpeg",
        user_id=request.user_id,
        require_liveness=effective_require_liveness,
        priority="high",
    )

    return {"job_id": job_id, "status": "pending"}


@router.post("/verify_async_file")
async def verify_async(
    http_request: Request,
    file: UploadFile = File(...),
    user_id: Optional[str] = Query(None),
    require_liveness: bool = Query(False),
    priority: str = Query("high"),
    db: AsyncSession = Depends(get_db),
):
    """Production async verify: file -> MinIO -> queue -> worker."""
    # Active-challenge gate допуска: multipart-путь не несёт liveness_token,
    # поэтому при LIVENESS_ACTIVE_REQUIRED=true и require_liveness=true — сразу
    # 403 с направлением на /verify_base64 (active proof работает там). До
    # резервирования slot'а, чтобы не резервировать чтобы тут же сбросить.
    if require_liveness and settings.LIVENESS_ACTIVE_REQUIRED:
        raise HTTPException(
            status_code=403,
            detail=(
                "active_liveness_required: multipart path does not carry liveness_token; "
                "use /api/v1/verify_base64 with liveness_mode=active + liveness_token"
            ),
        )
    slot_reserved = False
    slot_reserved = try_reserve_slot(max_queue_delay_ms=float(settings.BACKPRESSURE_MAX_QUEUE_DELAY_MS))
    if not slot_reserved:
        raise HTTPException(status_code=429, detail="Backpressure: queue_delay_sla")

    try:
        await asyncio.to_thread(RateLimiter.check, http_request, "verify_async", 5)

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid image format")

        image_bytes = await file.read()

        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=400, detail="Image too large")

        job_id = get_request_id(http_request)
        request_received_time = time.time()
        safe_filename = Path(file.filename or "image.jpg").name
        object_name = f"verify/{job_id}/{safe_filename}"

        await _enqueue_verify_job(
            db=db,
            job_id=job_id,
            request_received_time=request_received_time,
            image_bytes=image_bytes,
            object_name=object_name,
            content_type=file.content_type or "image/jpeg",
            user_id=user_id,
            require_liveness=require_liveness,
            priority=priority,
        )

        return {
            "job_id": job_id,
            "status": "pending",
        }
    finally:
        if slot_reserved:
            decrement_active()


@router.post("/verify_async_base64")
async def verify_async_base64(
    request: VerifyRequest,
    http_request: Request,
    priority: str = Query("high"),
    db: AsyncSession = Depends(get_db),
):
    """Async verify via queue/workers for real load."""
    # Active liveness: валидируем+consumes token ДО backpressure-логики (single-use).
    # NOTE: async-результат liveness_passed=True не выставляется воркером (active-proof
    #已知 только роуту) — для онлайн-контроля доступа используется sync fast-path.
    effective_require_liveness, active_proven = _resolve_liveness(request)

    queue_delay = get_system_load()
    mode = get_backpressure_mode(queue_delay)
    require_liveness = effective_require_liveness
    inflight = current_active_requests()
    dynamic_limit = get_inflight_limit(queue_delay)

    if mode == "degrade":
        require_liveness = False

    if mode == "shed" and should_drop_request(mode):
        raise HTTPException(status_code=429, detail="Backpressure: shed")

    if inflight >= dynamic_limit:
        if mode == "normal":
            logger.warning(f"INFLIGHT REJECT: {inflight}/{dynamic_limit}")
            raise HTTPException(status_code=429, detail="inflight limit")
        else:
            # in degrade/shed we give the request a chance to proceed
            pass

    slot_reserved = False
    slot_reserved = try_reserve_slot(max_queue_delay_ms=float(settings.BACKPRESSURE_MAX_QUEUE_DELAY_MS))
    if not slot_reserved:
        if mode == "normal":
            raise HTTPException(status_code=429, detail="Backpressure: queue")
        else:
            # in degrade/shed we give the request a chance to proceed
            pass

    try:
        await asyncio.to_thread(RateLimiter.check, http_request, "verify_async", 5)

        try:
            image_bytes = base64.b64decode(request.image)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64")

        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=400, detail="Image too large")

        job_id = get_request_id(http_request)
        request_received_time = time.time()
        object_name = f"verify/{job_id}/legacy.jpg"

        await _enqueue_verify_job(
            db=db,
            job_id=job_id,
            request_received_time=request_received_time,
            image_bytes=image_bytes,
            object_name=object_name,
            content_type="image/jpeg",
            user_id=request.user_id,
            require_liveness=require_liveness,
            priority=priority,
        )

        return {
            "job_id": job_id,
            "status": "pending",
        }
    finally:
        if slot_reserved:
            decrement_active()


@router.get("/verify_result/{job_id}")
async def get_verify_result(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    cached = redis_client.get(f"job:{job_id}")
    if cached:
        try:
            if isinstance(cached, bytes):
                cached = cached.decode("utf-8")
            return json.loads(cached)
        except Exception:
            pass

    job_repo = VerificationJobRepository(db)
    job = await job_repo.get_by_id(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.status.value if hasattr(job.status, "value") else job.status

    if status in ("pending", "processing"):
        return {
            "job_id": job_id,
            "status": status,
            "ready": False,
        }

    return {
        "job_id": job_id,
        "status": status,
        "result": job.result,
        "error": job.error,
        "ready": status in ("done", "failed"),
    }
