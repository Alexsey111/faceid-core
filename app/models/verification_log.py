# verification_log.py - Модель лога верификации

from sqlalchemy import Column, Integer, Float, Boolean, DateTime, ForeignKey, String
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

    # Клиентский IP для audit (ТЗ-схема: request_ip). Хранится как String(45)
    # (IPv4/IPv6 вмещается) — переносимее postgres-only INET; валидация формата
    # делается при извлечении (extract_client_ip), а не на уровне БД.
    # Пробрасывается через contextvar (app/core/request_context.py) — audit E2.
    request_ip = Column(String(45), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="verification_logs")