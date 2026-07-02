# test_liveness_challenge_ws.py — integration-тест active-liveness WS-эндпоинта.
#
# Поднимает FastAPI с liveness_challenge-роутером, эмулирует Redis + ML-компоненты
# (detector/landmarker/passive) фикстурами. Покрывает:
#   1) /init выдаёт challenge_id + ws_token + actions;
#   2) WS-стрим статических кадров → is_live=False (anti-cutout: фото не выполняет
#      действия), liveness_token не выдаётся;
#   3) happy-path (verify_challenge_stream мокнут в is_live=True) → выдаётся
#      liveness_token, он single-use (consume дважды → второе False);
#   4) single-use challenge: повторный стрим тем же challenge_id → close 4410.
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.routes.liveness_challenge as route
import app.services.liveness_token as lt
from app.core.config import settings
from app.ml.liveness.challenge import ChallengeResult

# WS-стрим поднимается in-process (TestClient) с FakeRedis и фейковым ML —
# внешняя БД/Redis не нужны → маркер 'unit' пропускает alembic/flushdb в conftest.
pytestmark = pytest.mark.unit


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def setex(self, key: str, value: str, ttl: int = 300) -> None:
        self.store[key] = value

    def set_if_absent(self, key: str, value: str, ttl: int = 300) -> bool:
        if key in self.store:
            return False
        self.store[key] = value
        return True


class _FakeDetector:
    """Один детект: bbox + 5pt landmarks (симметричное лицо → yaw≈0)."""

    def detect(self, image):
        return [{
            "bbox": [50.0, 50.0, 150.0, 150.0],
            "landmarks": [[75.0, 80.0], [125.0, 80.0], [100.0, 100.0],
                          [85.0, 130.0], [115.0, 130.0]],
            "confidence": 0.95,
        }]


class _FakeLandmarker:
    def get(self, image, bbox):
        return None  # EAR недоступен → blink не детектим


class _FakePassive:
    def predict(self, image, bbox):
        return 0.99, True


def _jpeg_bytes() -> bytes:
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def app_client(monkeypatch):
    # один FakeRedis для challenge-состояния AND liveness_token (token виден в consume)
    fake_redis = _FakeRedis()
    monkeypatch.setattr(route, "redis_client", fake_redis)
    monkeypatch.setattr(lt, "redis_client", fake_redis)
    monkeypatch.setattr(route, "_get_ml",
                        lambda: (_FakeDetector(), _FakeLandmarker(), _FakePassive()))
    monkeypatch.setattr(settings, "LIVENESS_ENABLED", True)
    monkeypatch.setattr(settings, "LIVENESS_ACTIVE_ENABLED", True)

    app = FastAPI()
    app.include_router(route.router)
    # сбрасываем ленивый _ML на случай загрязнения от прошлых тестов
    monkeypatch.setattr(route, "_ML", None)
    with TestClient(app) as client:
        yield client, fake_redis


def _init(client) -> dict:
    resp = client.post("/liveness/challenge/init")
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_init_returns_challenge_and_ws_url(app_client):
    client, _ = app_client
    data = _init(client)
    assert "challenge_id" in data and "ws_token" in data and "actions" in data
    assert data["ws_url"].startswith("/liveness/challenge/stream?")
    assert 1 <= len(data["actions"]) <= settings.LIVENESS_CHALLENGE_ACTIONS


def test_stream_static_frames_not_live_no_token(app_client):
    """Статические кадры (фото) не выполняют действия → is_live=False, token=None."""
    client, _ = app_client
    init_data = _init(client)
    cid, ws_token = init_data["challenge_id"], init_data["ws_token"]

    with client.websocket_connect(
        f"/liveness/challenge/stream?challenge_id={cid}&ws_token={ws_token}"
    ) as ws:
        challenge_msg = ws.receive_json()
        assert challenge_msg["type"] == "challenge"
        for _ in range(8):
            ws.send_bytes(_jpeg_bytes())
        ws.send_text(json.dumps({"cmd": "done"}))
        result = ws.receive_json()

    assert result["type"] == "result"
    assert result["is_live"] is False, "статичное фото не должно пройти challenge"
    assert result["liveness_token"] is None
    assert result["consistency_ok"] is True  # трек стабилен — провал по действиям


def test_stream_happy_path_issues_single_use_token(app_client, monkeypatch):
    """verify_challenge_stream→is_live=True → liveness_token выдаётся и single-use."""
    client, fake_redis = app_client

    def _fake_verify(obs, actions, threshold=None):
        return ChallengeResult(
            is_live=True, confidence=1.0,
            actions_performed={a: True for a in actions},
            consistency_ok=True, n_frames=len(obs), reason="ok",
        )

    monkeypatch.setattr(route, "verify_challenge_stream", _fake_verify)

    init_data = _init(client)
    cid, ws_token = init_data["challenge_id"], init_data["ws_token"]

    with client.websocket_connect(
        f"/liveness/challenge/stream?challenge_id={cid}&ws_token={ws_token}"
    ) as ws:
        ws.receive_json()  # challenge
        ws.send_bytes(_jpeg_bytes())
        ws.send_text(json.dumps({"cmd": "done"}))
        result = ws.receive_json()

    assert result["type"] == "result"
    assert result["is_live"] is True
    token = result["liveness_token"]
    assert token is not None
    # single-use: первый consume ок, повторный — нет (anti-replay)
    assert lt.consume_liveness_token(token) is True
    assert lt.consume_liveness_token(token) is False


def test_challenge_is_single_use_after_stream(app_client, monkeypatch):
    """После стрима challenge.used=True → повторный WS тем же id закрывается (4410)."""
    client, _ = app_client
    monkeypatch.setattr(route, "verify_challenge_stream",
                        lambda o, a, threshold=None: ChallengeResult(
                            is_live=True, confidence=1.0, actions_performed={},
                            consistency_ok=True, n_frames=len(o), reason="ok"))

    init_data = _init(client)
    cid, ws_token = init_data["challenge_id"], init_data["ws_token"]

    with client.websocket_connect(
        f"/liveness/challenge/stream?challenge_id={cid}&ws_token={ws_token}"
    ) as ws:
        ws.receive_json()
        ws.send_bytes(_jpeg_bytes())
        ws.send_text(json.dumps({"cmd": "done"}))
        ws.receive_json()

    # повторный стрим тем же challenge_id — должен быть отвергнут
    with pytest.raises(Exception):
        with client.websocket_connect(
            f"/liveness/challenge/stream?challenge_id={cid}&ws_token={ws_token}"
        ) as ws2:
            ws2.receive_json()  # сервер закрывает до challenge-сообщения