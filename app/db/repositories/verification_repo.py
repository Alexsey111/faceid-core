from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.verification_log import VerificationLog
from app.monitoring.db_metrics import timed_db_call


class VerificationRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_log(
        self,
        user_id: int | None,
        similarity: float,
        success: bool,
        margin: float | None = None,
        liveness_score: float | None = None,
        is_genuine: bool | None = None,
        commit: bool = True,
    ) -> Optional[VerificationLog]:
        if user_id is None:
            return None

        record = VerificationLog(
            user_id=user_id,
            similarity=similarity,
            margin=margin,
            liveness_score=liveness_score,
            is_genuine=is_genuine,
            result=success,
        )

        self.db.add(record)

        try:
            await timed_db_call(
                self.db.flush(),
                "verification_repo.create_log.flush",
            )
            if commit:
                await timed_db_call(
                    self.db.commit(),
                    "verification_repo.create_log.commit",
                )
        except IntegrityError:
            await self.db.rollback()
            return None

        return record
