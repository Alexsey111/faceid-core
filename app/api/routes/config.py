# app/api/routes/config.py — read-only публичные пороги для демо-GUI.
#
# Назначение: дать клиенту (demo/app.js) значения порогов match/liveness для
# наглядной интерпретации ответа /verify (client-side слайдер, индикаторы).
# Production-логику порогов НЕ мутирует — только отражает текущие settings.
# Секреты (SECRET_KEY, AES_SECRET_KEY, BIOMETRY_AES_KEY_B64, JWT_SECRET,
# API_KEYS) намеренно НЕ отдаются.
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import require_auth
from app.core.config import settings

router = APIRouter()


class PublicConfig(BaseModel):
    """Подмножество настроек, безопасное для отдачи клиенту демо-GUI."""

    FACE_MATCH_THRESHOLD: float        # config.py:247 — порог допуска (match)
    LIVENESS_THRESHOLD: float          # config.py:214 — порог passive liveness
    LIVENESS_ENABLED: bool             # config.py:210 — passive /liveness вкл/выкл
    LIVENESS_ACTIVE_ENABLED: bool      # config.py:223 — active challenge (WS) вкл/выкл
    LIVENESS_ACTIVE_REQUIRED: bool     # config.py:231 — обязательный active gate допуска
    QUALITY_GATE_MODE: str             # config.py:170 — hard | soft | off


@router.get("/config")
async def get_public_config(_auth: dict = Depends(require_auth)) -> dict:
    """Публичные пороги для демо-GUI. Production-логику НЕ мутирует.

    При AUTH_ENABLED=false (демо/dev) require_auth коротко замыкается —
    эндпоинт открыт, как и остальные /api/v1.
    """
    return PublicConfig(
        FACE_MATCH_THRESHOLD=settings.FACE_MATCH_THRESHOLD,
        LIVENESS_THRESHOLD=settings.LIVENESS_THRESHOLD,
        LIVENESS_ENABLED=settings.LIVENESS_ENABLED,
        LIVENESS_ACTIVE_ENABLED=settings.LIVENESS_ACTIVE_ENABLED,
        LIVENESS_ACTIVE_REQUIRED=settings.LIVENESS_ACTIVE_REQUIRED,
        QUALITY_GATE_MODE=settings.QUALITY_GATE_MODE,
    ).model_dump()