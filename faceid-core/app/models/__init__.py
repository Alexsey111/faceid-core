# __init__.py

# app/models/__init__.py
from app.db.base import Base

# ВАЖНО: импортируйте сначала User, потом Embedding
from app.models.user import User
from app.models.embedding import Embedding
from app.models.verification_log import VerificationLog  # если есть

# Экспортируйте все модели
__all__ = ["Base", "User", "Embedding", "VerificationLog"]
