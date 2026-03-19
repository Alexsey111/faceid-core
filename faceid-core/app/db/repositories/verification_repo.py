# verification_repo.py - Репозиторий логов верификации

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.verification_log import VerificationLog
from app.models.user import User


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
        is_genuine: bool | None = None
    ) -> Optional[VerificationLog]:
        """
        Create a verification log entry.

        Args:
            user_id: Matched user ID
            similarity: Cosine similarity score
            success: Whether verification was successful
            margin: Difference between top-1 and top-2 similarity
            liveness_score: Liveness detection score
            is_genuine: Whether matched user matches expected user
        """
        # Skip if user_id is None or doesn't exist in users table
        if user_id is None:
            return None

        # Check if user exists
        result = await self.db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            # User doesn't exist, skip logging
            return None

        record = VerificationLog(
            user_id=user_id,
            similarity=similarity,
            margin=margin,
            liveness_score=liveness_score,
            is_genuine=is_genuine,
            result=success
        )

        self.db.add(record)
        await self.db.flush()
        await self.db.commit()

        return record