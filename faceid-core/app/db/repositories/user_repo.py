# user_repo.py - Репозиторий пользователей

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User


class UserRepository:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_or_create(self, external_id: str) -> User:
        """Get existing user or create new one."""
        query = select(User).where(User.external_id == external_id)
        result = await self.db.execute(query)
        user = result.scalar_one_or_none()

        if user:
            return user

        # Create new user
        user = User(external_id=external_id)
        self.db.add(user)
        await self.db.flush()
        await self.db.refresh(user)
        return user

    async def get_by_id(self, user_id: int) -> User | None:
        query = select(User).where(User.id == user_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_external_id(self, external_id: str) -> User | None:
        query = select(User).where(User.external_id == external_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
