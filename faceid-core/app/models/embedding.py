# embedding.py - Модель эмбеддинга

import numpy as np

from datetime import datetime
from sqlalchemy import ForeignKey, DateTime, LargeBinary
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from app.db.base import Base
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from app.models.user import User


class Embedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        index=True,
        nullable=False
    )

    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(512),
        nullable=True
    )

    encrypted_embedding: Mapped[bytes] = mapped_column(
        LargeBinary,
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="embeddings",
        lazy="selectin"
    )

    @property
    def decrypted_vector(self) -> np.ndarray:
        from app.core.crypto import decrypt_vector
        return decrypt_vector(self.encrypted_embedding)
