# user.py - Модель пользователя

from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.base import Base


class User(Base):

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    external_id = Column(String, unique=True, index=True, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    embeddings = relationship("Embedding", back_populates="user")
    verification_logs = relationship("VerificationLog", back_populates="user")