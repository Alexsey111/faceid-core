# embedding.py - Модель эмбеддинга

import numpy as np

from datetime import datetime
from sqlalchemy import ForeignKey, DateTime, LargeBinary
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

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

    # Биометрия хранится ТОЛЬКО зашифрованной (AES-256-GCM). Plaintext-копия
    # (pgvector-колонка `embedding`) удалена — ТЗ требует хранения шаблона
    # исключительно в зашифрованном виде. Поиск ведётся через FAISS (строится
    # из encrypted при старте); SQL-fallback — decrypt-all + numpy-cosine.
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
