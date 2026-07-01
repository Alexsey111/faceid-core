# tests/test_auth.py — P0.2: аутентификация эндпоинтов (JWT + X-API-Key).
#
# Юнит-тесты: поднимают изолированное FastAPI-приложение с require_auth,
# без БД/Redis/инфры (маркер `unit` → conftest пропускает миграции).

import pytest
import jwt as pyjwt
from datetime import datetime, timedelta, timezone
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api.deps import require_auth
from app.core.config import settings

pytestmark = pytest.mark.unit

_TEST_SECRET = "test-jwt-secret"
_TEST_API_KEYS = "svc-key-aaa,svc-key-bbb"


def _build_app() -> FastAPI:
    """Изолированное приложение: один защищённый и один открытый эндпоинт."""
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/protected", dependencies=[Depends(require_auth)])
    async def protected() -> dict:
        return {"ok": True}

    @app.post("/protected-post", dependencies=[Depends(require_auth)])
    async def protected_post() -> dict:
        return {"ok": True}

    return app


@pytest.fixture()
def auth_enabled(monkeypatch):
    """Включает аутентификацию и задаёт тестовые секреты."""
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "JWT_SECRET", _TEST_SECRET)
    monkeypatch.setattr(settings, "JWT_ALG", "HS256")
    monkeypatch.setattr(settings, "JWT_ISSUER", None)
    monkeypatch.setattr(settings, "JWT_AUDIENCE", None)
    monkeypatch.setattr(settings, "API_KEYS", _TEST_API_KEYS)


@pytest.fixture()
def client(auth_enabled):
    return TestClient(_build_app())


def _mint_token(*, exp_delta_s: int = 60, sub: str = "user-1",
                issuer: str | None = None, audience: str | None = None,
                secret: str = _TEST_SECRET) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": sub, "exp": now + timedelta(seconds=exp_delta_s)}
    if issuer:
        payload["iss"] = issuer
    if audience:
        payload["aud"] = audience
    return pyjwt.encode(payload, secret, algorithm="HS256")


# ---- AUTH_ENABLED=False ---------------------------------------------------

def test_auth_disabled_allows_anonymous(monkeypatch):
    """При AUTH_ENABLED=False защищённый эндпоинт доступен без учётных данных."""
    monkeypatch.setattr(settings, "AUTH_ENABLED", False)
    app = _build_app()
    with TestClient(app) as c:
        resp = c.get("/protected")
    assert resp.status_code == 200, resp.text


# ---- Health остаётся открытым ----------------------------------------------

def test_health_open_without_auth(client):
    """/health не защищён — доступен без заголовков даже при AUTH_ENABLED=True."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---- Отсутствие учётных данных --------------------------------------------

def test_protected_no_credentials_401(client):
    resp = client.get("/protected")
    assert resp.status_code == 401


# ---- X-API-Key -------------------------------------------------------------

def test_protected_valid_api_key_200(client):
    resp = client.get("/protected", headers={"X-API-Key": "svc-key-aaa"})
    assert resp.status_code == 200, resp.text


def test_protected_invalid_api_key_401(client):
    resp = client.get("/protected", headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401


def test_protected_second_api_key_200(client):
    # Проверка, что CSV-парсинг даёт множество (оба ключа валидны).
    resp = client.get("/protected", headers={"X-API-Key": "svc-key-bbb"})
    assert resp.status_code == 200


# ---- JWT ------------------------------------------------------------------

def test_protected_valid_jwt_200(client):
    token = _mint_token()
    resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text


def test_protected_invalid_jwt_401(client):
    resp = client.get("/protected", headers={"Authorization": "Bearer not.a.jwt"})
    assert resp.status_code == 401


def test_protected_expired_jwt_401(client):
    # exp в прошлом.
    token = _mint_token(exp_delta_s=-10)
    resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


def test_protected_wrong_secret_jwt_401(client):
    token = _mint_token(secret="another-secret")
    resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


def test_protected_jwt_without_exp_401(client):
    # options={"require": ["exp"]} → токен без exp отклоняется.
    payload = {"sub": "user-1"}
    token = pyjwt.encode(payload, _TEST_SECRET, algorithm="HS256")
    resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


def test_protected_post_valid_jwt_200(client):
    token = _mint_token()
    resp = client.post("/protected-post", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text


# ---- issuer/audience проверяются, если заданы -----------------------------

def test_protected_jwt_wrong_issuer_401(monkeypatch, client):
    monkeypatch.setattr(settings, "JWT_ISSUER", "faceid")
    token = _mint_token(issuer="other-issuer")
    resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


def test_protected_jwt_correct_issuer_200(monkeypatch, client):
    monkeypatch.setattr(settings, "JWT_ISSUER", "faceid")
    token = _mint_token(issuer="faceid")
    resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text