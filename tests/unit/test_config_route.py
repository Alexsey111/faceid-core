# test_config_route.py — read-only GET /api/v1/config для демо-GUI.
# Маркер 'unit' → без DB/Redis. AUTH_ENABLED=false ставит conftest.
from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api.deps import require_auth
from app.api.routes.config import router as config_router

pytestmark = pytest.mark.unit

# Изолированное приложение: только config-роутер под require_auth (как в router.py).
# Не поднимаем app.main:app (его lifespan тянется к БД) — для unit-проверки контракта
# достаточно роутера в пустой FastAPI.
_AUTH = [Depends(require_auth)]


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(config_router, prefix="/api/v1", dependencies=_AUTH)
    return TestClient(app)


EXPECTED_FIELDS = {
    "FACE_MATCH_THRESHOLD",
    "LIVENESS_THRESHOLD",
    "LIVENESS_ENABLED",
    "LIVENESS_ACTIVE_ENABLED",
    "LIVENESS_ACTIVE_REQUIRED",
    "QUALITY_GATE_MODE",
}

# Секреты, которые НЕ должны попасть в ответ (config.py:263-278).
SECRET_FIELDS = {
    "SECRET_KEY",
    "AES_SECRET_KEY",
    "BIOMETRY_AES_KEY_B64",
    "JWT_SECRET",
    "API_KEYS",
    "MINIO_SECRET_KEY",
}


def test_config_returns_exactly_expected_fields():
    """GET /api/v1/config отдаёт ровно 6 публичных порогов — ничего лишнего."""
    with _client() as c:
        resp = c.get("/api/v1/config")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == EXPECTED_FIELDS


def test_config_does_not_leak_secrets():
    """Ответ НЕ содержит секретных ключей (AES/JWT/API_KEYS/MinIO)."""
    with _client() as c:
        data = c.get("/api/v1/config").json()
    for secret in SECRET_FIELDS:
        assert secret not in data, f"секрет {secret} утёк в /config"


def test_config_field_types():
    """Типы полей корректны (float/bool/str) и совпадают с контрактом."""
    with _client() as c:
        data = c.get("/api/v1/config").json()
    assert isinstance(data["FACE_MATCH_THRESHOLD"], float)
    assert isinstance(data["LIVENESS_THRESHOLD"], float)
    assert isinstance(data["LIVENESS_ENABLED"], bool)
    assert isinstance(data["LIVENESS_ACTIVE_ENABLED"], bool)
    assert isinstance(data["LIVENESS_ACTIVE_REQUIRED"], bool)
    assert isinstance(data["QUALITY_GATE_MODE"], str)
    assert data["QUALITY_GATE_MODE"] in ("hard", "soft", "off")


def test_config_open_without_auth_when_disabled():
    """При AUTH_ENABLED=false эндпоинт открыт без JWT/X-API-Key (условие демо)."""
    # conftest уже выставил AUTH_ENABLED=false.
    with _client() as c:
        resp = c.get("/api/v1/config")
    assert resp.status_code == 200