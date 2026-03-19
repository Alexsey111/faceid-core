# verification_log.py - Модель лога верификации

from sqlalchemy import Column, Integer, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.base import Base


class VerificationLog(Base):

    __tablename__ = "verification_logs"

    id = Column(Integer, primary_key=True)

    user_id = Column(Integer, ForeignKey("users.id"))

    similarity = Column(Float)

    margin = Column(Float, nullable=True)

    liveness_score = Column(Float, nullable=True)

    is_genuine = Column(Boolean, nullable=True)

    result = Column(Boolean)

    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="verification_logs")