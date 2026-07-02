# test_liveness_token.py — unit-тесты single-use liveness_token (proof для /verify).
#
# FakeRedis эмулирует app.infrastructure.redis_client.RedisClient: get / setex /
# set_if_absent (SETNX). Проверяем: issue → validate True; consume первый True,
# повторный consume False (single-use anti-replay); expired/None token → False.
from __future__ import annotations

import json
import pytest

import app.services.liveness_token as lt
from app.core.config import settings

_unit = pytest.mark.unit


class _FakeRedis:
    """Эмулирует RedisClient: get / setex(key,value,ttl) / set_if_absent(key,value,ttl)."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def setex(self, key: str, value: str, ttl: int = 300) -> None:
        self.store[key] = value  # ttl игнорируем (unit-тест)

    def set_if_absent(self, key: str, value: str, ttl: int = 300) -> bool:
        if key in self.store:
            return False
        self.store[key] = value
        return True


@pytest.fixture
def fake_redis(monkeypatch):
    fr = _FakeRedis()
    monkeypatch.setattr(lt, "redis_client", fr)
    return fr


@_unit
def test_issue_then_validate_true(fake_redis):
    token = lt.issue_liveness_token("ch-1")
    assert isinstance(token, str) and len(token) > 0
    assert lt.validate_liveness_token(token) is True


@_unit
def test_consume_is_single_use(fake_redis):
    token = lt.issue_liveness_token("ch-2")
    assert lt.consume_liveness_token(token) is True, "первый consume валидного token → True"
    # повторный consume того же token → False (anti-replay)
    assert lt.consume_liveness_token(token) is False
    assert lt.validate_liveness_token(token) is False, "после consume validate тоже False"


@_unit
def test_consume_unknown_token_is_false(fake_redis):
    assert lt.consume_liveness_token("nonexistent-token") is False
    assert lt.consume_liveness_token(None) is False


@_unit
def test_validate_none_and_empty_is_false(fake_redis):
    assert lt.validate_liveness_token(None) is False
    assert lt.validate_liveness_token("") is False


@_unit
def test_expired_token_is_false(fake_redis):
    """Симуляция истечения TTL: удаляем key из store → validate/consume False."""
    token = lt.issue_liveness_token("ch-3")
    # эмулируем истечение
    del fake_redis.store[lt._LVKEY + token]
    assert lt.validate_liveness_token(token) is False
    assert lt.consume_liveness_token(token) is False


@_unit
def test_token_carries_challenge_id(fake_redis):
    token = lt.issue_liveness_token("ch-abc")
    raw = fake_redis.store[lt._LVKEY + token]
    payload = json.loads(raw)
    assert payload["challenge_id"] == "ch-abc"