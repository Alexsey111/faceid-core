# app/api/routes/verify.py

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.verification_job_repo import VerificationJobRepository
from app.db.session import get_db
from app.infrastructure.minio_client import MinioClient
from app.infrastructure.redis_client import redis_client
from app.models.verification_job import JobStatus
from app.schemas.verify import VerifyRequest, VerifyResponse
from app.services.backpressure import decrement_active, try_reserve_slot
from app.services.rate_limiter import RateLimiter
from app.services.verification_service_factory import get_verification_service
from app.workers.tasks.verify_task import verify_task

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE = 5 * 1024 * 1024


async def _enqueue_verify_job(
    *,
    db: AsyncSession,
    job_id: str,
    image_bytes: bytes,
    object_name: str,
    content_type: str,
    user_id: str | None,
    require_liveness: bool,
) -> None:
    job_repo = VerificationJobRepository(db)
    await job_repo.create(job_id=job_id, status=JobStatus.pending)
    await db.commit()

    minio_client = MinioClient()

    try:
        await asyncio.to_thread(
            minio_client.upload_image,
            object_name,
            image_bytes,
            content_type,
        )
        verify_task.delay(
            job_id=job_id,
            image_url=object_name,
            user_id=user_id,
            require_liveness=require_liveness,
        )
    except Exception as exc:
        await job_repo.update(job_id, status=JobStatus.failed, error=str(exc))
        await db.commit()
        raise HTTPException(status_code=500, detail="Failed to enqueue verify job")

    logger.info("verify_async_enqueued", extra={"job_id": job_id, "image_url": object_name})


@router.post("/verify", response_model=VerifyResponse)
async def verify_file(
    http_request: Request,
    file: UploadFile = File(...),
    user_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Verify face against reference via multipart/form-data."""
    RateLimiter.check(http_request, "verify", limit=10)

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

    return result


@router.post("/verify_base64", response_model=VerifyResponse)
async def verify_base64(
    request: VerifyRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Verify face against reference via JSON with base64 image."""
    RateLimiter.check(http_request, "verify", limit=10)

    image_bytes = base64.b64decode(request.image)

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="Image too large")

    service = get_verification_service(db)

    result = await service.verify_face(
        image_bytes,
        user_id=request.user_id,
        require_liveness=request.require_liveness,
    )

    return result


@router.post("/verify_async")
async def verify_async(
    http_request: Request,
    file: UploadFile = File(...),
    user_id: Optional[str] = Query(None),
    require_liveness: bool = Query(False),
    db: AsyncSession = Depends(get_db),
):
    """Production async verify: file -> MinIO -> queue -> worker."""
    if not try_reserve_slot():
        raise HTTPException(status_code=429, detail="Backpressure: active_limit")

    try:
        await asyncio.to_thread(RateLimiter.check, http_request, "verify_async", 5)

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid image format")

        image_bytes = await file.read()

        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=400, detail="Image too large")

        job_id = str(uuid4())
        safe_filename = Path(file.filename or "image.jpg").name
        object_name = f"verify/{job_id}/{safe_filename}"

        await _enqueue_verify_job(
            db=db,
            job_id=job_id,
            image_bytes=image_bytes,
            object_name=object_name,
            content_type=file.content_type or "image/jpeg",
            user_id=user_id,
            require_liveness=require_liveness,
        )

        return {
            "job_id": job_id,
            "status": "pending",
        }
    except Exception:
        decrement_active()
        raise


@router.post("/verify_async_base64")
async def verify_async_base64(
    request: VerifyRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Legacy async verify path that still accepts JSON base64."""
    if not try_reserve_slot():
        raise HTTPException(status_code=429, detail="Backpressure: active_limit")

    try:
        await asyncio.to_thread(RateLimiter.check, http_request, "verify_async", 5)

        try:
            image_bytes = base64.b64decode(request.image)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64")

        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=400, detail="Image too large")

        job_id = str(uuid4())
        object_name = f"verify/{job_id}/legacy.jpg"

        await _enqueue_verify_job(
            db=db,
            job_id=job_id,
            image_bytes=image_bytes,
            object_name=object_name,
            content_type="image/jpeg",
            user_id=request.user_id,
            require_liveness=request.require_liveness,
        )

        return {
            "job_id": job_id,
            "status": "pending",
        }
    except Exception:
        decrement_active()
        raise


@router.get("/verify_result/{job_id}")
async def get_verify_result(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    cached = redis_client.get(f"job:{job_id}")
    if cached:
        try:
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
