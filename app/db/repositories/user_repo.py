# user_repo.py - Репозиторий пользователей

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.monitoring.db_metrics import timed_db_call


class UserRepository:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_or_create(self, external_id: str) -> User:
        """Get existing user or create new one."""
        query = select(User).where(User.external_id == external_id)
        result = await timed_db_call(self.db.execute(query), "user_repo.get_or_create")
        user = result.scalar_one_or_none()

        if user:
            return user

        # Create new user
        user = User(external_id=external_id)
        self.db.add(user)
        await self.db.flush()
        await self.db.refresh(user)
        return user

    async def get_by_external_id(self, external_id: str) -> User | None:
        query = select(User).where(User.external_id == external_id)
        result = await timed_db_call(self.db.execute(query), "user_repo.get_by_external_id")
        return result.scalar_one_or_none()
