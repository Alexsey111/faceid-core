# test_api_contract.py — API-контракт: /api/v1 prefix, match_score алиас,
# confidence (high/medium/low), SIM_THRESHOLD выровнен с LOW_THRESHOLD.
# Маркер 'unit' → без DB/Redis.
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.verify import router as verify_router
from app.core.config import settings
from app.schemas.verify import VerifyResponse
from app.services.verification_service import _confidence_label

pytestmark = pytest.mark.unit


# --- confidence ---

def test_confidence_high_for_match():
    assert _confidence_label(0.87, "match") == "high"
    # граница HIGH_THRESHOLD (0.45) включительно → high
    assert _confidence_label(settings.HIGH_THRESHOLD, "match") == "high"


def test_confidence_medium_for_low_confidence_band():
    # 0.3 ≤ sim < 0.45 → medium (low_confidence band)
    assert _confidence_label(0.40, "low_confidence") == "medium"
    assert _confidence_label(settings.LOW_THRESHOLD, "low_confidence") == "medium"


def test_confidence_low_for_no_match():
    assert _confidence_label(0.1, "no_match") == "low"


def test_confidence_none_when_no_match_decision():
    # match не считался → None для не-decision статусов
    for status in ("spoof_detected", "quality_reject", "retry", "processing_failed"):
        assert _confidence_label(0.0, status) is None
    assert _confidence_label(None, "match") is None


# --- VerifyResponse: match_score алиас similarity ---

def test_verify_response_alias_similarity_to_match_score():
    resp = VerifyResponse(status="match", similarity=0.87, confidence="high")
    assert resp.match_score == 0.87
    assert resp.similarity == 0.87  # legacy поле сохранено


def test_verify_response_explicit_match_score_preserved():
    resp = VerifyResponse(status="match", match_score=0.9, similarity=0.87)
    assert resp.match_score == 0.9  # явное match_score не перетёрто similarity


def test_verify_response_no_similarity_match_score_none():
    resp = VerifyResponse(status="quality_reject", reason="image_blurry")
    assert resp.match_score is None
    assert resp.confidence is None


# --- /api/v1 prefix ---

def test_api_v1_prefix_mounted():
    # Минимальное приложение: verify-роутер под /api/v1 (как в app.main:create_app).
    app = FastAPI()
    app.include_router(verify_router, prefix="/api/v1")
    client = TestClient(app)
    # /verify без префикса → 404, /api/v1/verify → не 404 (403/422 — auth/validation,
    # но точно не 404). Проверяем именно наличие пути под префиксом.
    assert client.post("/verify", json={}).status_code == 404
    prefixed = client.post("/api/v1/verify", json={})
    assert prefixed.status_code != 404


# --- SIM_THRESHOLD выровнен с LOW_THRESHOLD ---

def test_sim_threshold_aligned_with_low_threshold():
    # Pre-filter поиска не должен срезать low_confidence-диапазон (0.3–0.45).
    # SIM_THRESHOLD == LOW_THRESHOLD (no_match boundary), а не 0.45.
    assert settings.SIM_THRESHOLD == settings.LOW_THRESHOLD == 0.30