# app/services/liveness_token.py — короткоживущий proof «liveness пройден» для /verify.
#
# После успешного challenge (/liveness/challenge/stream) выдаётся liveness_token,
# привязанный к challenge_id (single-use, TTL=LIVENESS_TOKEN_TTL_S). /verify с
# liveness_mode=active валидирует и consumes этот token — не запуская passive
# повторно. Single-use атомарен через Redis SETNX consumed-маркера (нет race
# между параллельными /verify с одним token).
from __future__ import annotations

import json
import logging
import uuid

from app.core.config import settings
from app.infrastructure.redis_client import redis_client

logger = logging.getLogger("liveness.token")

_LVKEY = "lvtoken:"           # состояние token (challenge_id)
_LVUSED = "lvtoken:used:"     # SETNX-маркер «consumed» (atomic single-use)


def issue_liveness_token(challenge_id: str) -> str:
    """Создать token после успешного challenge. Возвращает hex-строку."""
    token = uuid.uuid4().hex
    redis_client.setex(
        _LVKEY + token,
        json.dumps({"challenge_id": challenge_id}),
        ttl=settings.LIVENESS_TOKEN_TTL_S,
    )
    logger.info("liveness_token issued challenge_id=%s ttl=%ss", challenge_id, settings.LIVENESS_TOKEN_TTL_S)
    return token


def validate_liveness_token(token: str | None) -> bool:
    """True если token валиден и ещё не потреблён (без consume)."""
    if not token:
        return False
    raw = redis_client.get(_LVKEY + token)
    if raw is None:
        return False
    # consumed-маркер существует → уже использован
    if redis_client.get(_LVUSED + token) is not None:
        return False
    return True


def consume_liveness_token(token: str | None) -> bool:
    """Atomic single-use: пометить token потреблённым. True если был валиден.

    Использует SETNX consumed-маркера — первая транзакция выигрывает, повторные
    /verify с тем же token получат False (anti-replay).
    """
    if not token:
        return False
    raw = redis_client.get(_LVKEY + token)
    if raw is None:
        return False  # expired/не существует
    acquired = redis_client.set_if_absent(
        _LVUSED + token, "1", ttl=settings.LIVENESS_TOKEN_TTL_S
    )
    return bool(acquired)