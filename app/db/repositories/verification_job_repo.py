# verification_job_repo.py - Repository for verification jobs

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.verification_job import VerificationJob
from app.monitoring.db_metrics import timed_db_call


class VerificationJobRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self,
        job_id: str,
        status: str = "pending",
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> VerificationJob:
        job = VerificationJob(
            id=job_id,
            status=status,
            result=result,
            error=error,
        )
        self.db.add(job)
        await self.db.flush()
        return job

    async def get_by_id(self, job_id: str) -> VerificationJob | None:
        query = select(VerificationJob).where(VerificationJob.id == job_id)
        result = await timed_db_call(self.db.execute(query), "verification_job_repo.get_by_id")
        return result.scalar_one_or_none()

    async def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        from sqlalchemy import update

        values: dict[str, Any] = {}

        if status is not None:
            values["status"] = status
        if result is not None:
            values["result"] = result
        if error is not None:
            values["error"] = error

        if not values:
            return

        stmt = (
            update(VerificationJob)
            .where(VerificationJob.id == job_id)
            .values(**values)
        )
        db_result: CursorResult[Any] = await timed_db_call(
            self.db.execute(stmt),
            "verification_job_repo.update",
        )

        if db_result.rowcount == 0:
            raise LookupError("Job not found")
