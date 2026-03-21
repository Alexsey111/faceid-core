# workers/tasks/verify_task.py - Verification async job task

from __future__ import annotations

import asyncio
import logging

from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.verification_job_repo import VerificationJobRepository
from app.db.repositories.verification_repo import VerificationRepository
from app.db.session import AsyncSessionLocal
from app.infrastructure.minio_client import MinioClient
from app.models.verification_job import JobStatus
from app.services.verification_service import VerificationService
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


async def _process_verify_job(
    job_id: str,
    image_url: str,
    user_id: str | None,
    require_liveness: bool,
) -> None:
    async with AsyncSessionLocal() as db:
        job_repo = VerificationJobRepository(db)

        try:
            job = await job_repo.get_by_id(job_id)
            if job is None:
                raise LookupError("Job not found")

            if job.status in (JobStatus.done, JobStatus.failed):
                return

            await job_repo.update(job_id, status=JobStatus.processing, error=None)
            await db.commit()

            minio_client = MinioClient()
            try:
                image_bytes = minio_client.get_image(image_url)
            except Exception as exc:
                raise ValueError("Invalid image payload") from exc

            embedding_repo = EmbeddingRepository(db)
            verification_repo = VerificationRepository(db)
            service = VerificationService(embedding_repo, verification_repo)

            result = await service.verify_face(
                image_bytes=image_bytes,
                user_id=user_id,
                require_liveness=require_liveness,
            )

            await job_repo.update(
                job_id,
                status=JobStatus.done,
                result=result,
                error=None,
            )
            await db.commit()
        except LookupError:
            logger.error("verify job not found", extra={"job_id": job_id})
            return
        except Exception as exc:
            logger.exception("verify job failed", extra={"job_id": job_id})
            await db.rollback()
            try:
                await job_repo.update(
                    job_id,
                    status=JobStatus.failed,
                    error=str(exc),
                )
                await db.commit()
            except LookupError:
                logger.error("verify job not found while marking failed", extra={"job_id": job_id})


@celery_app.task(name="app.workers.tasks.verify_job", bind=False)
def verify_task(
    job_id: str,
    image_url: str,
    user_id: str | None = None,
    require_liveness: bool = False,
):
    asyncio.run(_process_verify_job(job_id, image_url, user_id, require_liveness))
