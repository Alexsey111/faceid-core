# test_verify_active_liveness.py — unit-тесты логики active-liveness в /verify.
#
# _resolve_liveness — точка интеграции challenge→verify: при liveness_mode="active"
# валидирует+consumes liveness_token (single-use, 403 при невалидном), отключает
# повторный passive (effective_require_liveness=False), ставит active_proven=True.
# Полный verify-pipeline не поднимается — тестируется только эта логика + token-сторона.
from __future__ import annotations

import pytest
from fastapi import HTTPException

import app.services.liveness_token as lt
from app.api.routes.verify import _resolve_liveness
from app.schemas.verify import VerifyRequest

_unit = pytest.mark.unit


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


@pytest.fixture
def fake_redis(monkeypatch):
    fr = _FakeRedis()
    monkeypatch.setattr(lt, "redis_client", fr)
    return fr


def _req(mode: str, token: str | None = None, require_liveness: bool = True) -> VerifyRequest:
    return VerifyRequest(
        image="",
        require_liveness=require_liveness,
        liveness_mode=mode,
        liveness_token=token,
    )


@_unit
def test_active_valid_token_consumes_and_disables_passive(fake_redis):
    token = lt.issue_liveness_token("ch-1")
    eff_req, active_proven = _resolve_liveness(_req("active", token))
    assert eff_req is False, "active → passive не запускается повторно"
    assert active_proven is True
    # single-use: повторный resolve тем же token → 403
    with pytest.raises(HTTPException) as exc:
        _resolve_liveness(_req("active", token))
    assert exc.value.status_code == 403


@_unit
def test_active_invalid_token_raises_403(fake_redis):
    with pytest.raises(HTTPException) as exc:
        _resolve_liveness(_req("active", "bogus"))
    assert exc.value.status_code == 403


@_unit
def test_active_none_token_raises_403(fake_redis):
    with pytest.raises(HTTPException) as exc:
        _resolve_liveness(_req("active", None))
    assert exc.value.status_code == 403


@_unit
def test_active_mode_is_case_insensitive(fake_redis):
    token = lt.issue_liveness_token("ch-2")
    eff_req, active_proven = _resolve_liveness(_req("  ACTIVE ", token))
    assert active_proven is True and eff_req is False


@_unit
def test_passive_mode_passes_through_and_does_not_consume(fake_redis):
    # passive — token не трогается, require_liveness пробрасывается как есть
    eff_req, active_proven = _resolve_liveness(_req("passive", require_liveness=True))
    assert eff_req is True
    assert active_proven is False
    # неизвестный режим трактуется как passive (не падает, не потребляет token)
    eff_req2, active_proven2 = _resolve_liveness(_req("weird", require_liveness=False))
    assert eff_req2 is False and active_proven2 is False