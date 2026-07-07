# test_upload_validation.py — фикс пункта 7 аудита upload-gaps.
# Покрывает: (1) enroll_face поднимает ValueError при quality_reject/retry/no_face
# (раньше KeyError → 500; теперь 400 со структурированным reason);
# (2) /upload и /upload_base64 валидируют размер (≤5MB) и MIME.
# Маркер 'unit' → без DB/Redis (моки).
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api.routes import upload as upload_route
from app.api.routes.upload import MAX_IMAGE_SIZE, router as upload_router
from app.db.session import get_db
from app.services.embedding_service import EmbeddingService
from app.services.rate_limiter import RateLimiter

pytestmark = pytest.mark.unit


# ---------- 1. enroll_face: quality_reject/retry → ValueError (fix баги) ----------

def _service_with_pipeline_result(result: dict[str, Any]) -> EmbeddingService:
    """EmbeddingService с замоканным pipeline (без загрузки ONNX-моделей)."""
    repo = MagicMock()
    repo.create_embedding = AsyncMock(return_value=MagicMock(id=42))
    svc = EmbeddingService(repo, user_repo=None)
    svc.pipeline = MagicMock()
    svc.pipeline.process = MagicMock(return_value=result)
    svc.search_service = MagicMock()
    svc.search_service.invalidate_cache = AsyncMock()
    return svc


@pytest.mark.asyncio
async def test_enroll_quality_reject_raises_valueerror():
    """quality_reject из pipeline → ValueError (HTTP 400), НЕ KeyError/500."""
    svc = _service_with_pipeline_result({
        "status": "quality_reject",
        "quality_reason": "image_blurry",
        "quality_details": {"blur_score": 12.3},
    })
    with pytest.raises(ValueError) as exc:
        await svc.enroll_face(user_id=1, image_bytes=b"\x00")
    assert "enroll_failed" in str(exc.value)
    assert "image_blurry" in str(exc.value)


@pytest.mark.asyncio
async def test_enroll_retry_raises_valueerror():
    """retry (remove_occlusion) из pipeline → ValueError (HTTP 400)."""
    svc = _service_with_pipeline_result({
        "status": "retry",
        "quality_reason": "remove_occlusion",
        "quality_details": {"occlusion_flags": {"mask_detected": True}},
    })
    with pytest.raises(ValueError):
        await svc.enroll_face(user_id=1, image_bytes=b"\x00")


@pytest.mark.asyncio
async def test_enroll_no_face_raises_valueerror():
    """no_face из pipeline → ValueError (HTTP 400)."""
    svc = _service_with_pipeline_result({"status": "no_face"})
    with pytest.raises(ValueError):
        await svc.enroll_face(user_id=1, image_bytes=b"\x00")


@pytest.mark.asyncio
async def test_enroll_ok_returns_embedding_id():
    """status='ok' → нормальный enroll, возвращает embedding_id (регрессия fix)."""
    import numpy as np
    svc = _service_with_pipeline_result({
        "status": "ok",
        "embedding": np.zeros(512, dtype=np.float32),
    })
    res = await svc.enroll_face(user_id=1, image_bytes=b"\x00")
    assert res["embedding_id"] == 42


# ---------- 2. /upload, /upload_base64: валидация размера и MIME ----------

def _client_with_mock_deps(monkeypatch) -> TestClient:
    """TestClient с замоченными get_db и RateLimiter (без Redis/Postgres)."""
    # RateLimiter.check лезет в redis — глушим.
    monkeypatch.setattr(RateLimiter, "check", staticmethod(lambda *a, **k: None))
    # get_db отдаёт MagicMock-сессию; EmbeddingService всё равно не дойдём до DB
    # при срабатывании валидации (size/MIME checks раньше enroll_face).
    async def _fake_db():
        yield MagicMock()
    app = FastAPI()
    app.include_router(upload_router, prefix="/api/v1")
    app.dependency_overrides[get_db] = _fake_db
    return TestClient(app)


def test_upload_base64_rejects_oversize(monkeypatch):
    """base64 > 5MB → 400 (валидация размера до enroll)."""
    import base64
    big = base64.b64encode(b"\x00" * (MAX_IMAGE_SIZE + 1)).decode()
    with _client_with_mock_deps(monkeypatch) as c:
        resp = c.post("/api/v1/upload_base64", json={"user_id": "1", "image": big})
    assert resp.status_code == 400
    assert "too large" in resp.json()["detail"].lower()


def test_upload_base64_rejects_empty(monkeypatch):
    """пустой image → 400."""
    with _client_with_mock_deps(monkeypatch) as c:
        resp = c.post("/api/v1/upload_base64", json={"user_id": "1", "image": ""})
    assert resp.status_code == 400


def test_upload_multipart_rejects_non_image_mime(monkeypatch):
    """/upload с не-image content_type → 400."""
    with _client_with_mock_deps(monkeypatch) as c:
        resp = c.post(
            "/api/v1/upload?user_id=1",
            files={"file": ("f.txt", b"x", "text/plain")},
        )
    assert resp.status_code == 400
    assert "image" in resp.json()["detail"].lower()


def test_upload_multipart_rejects_oversize(monkeypatch):
    """/upload с изображением > 5MB → 413."""
    big = b"\x00" * (MAX_IMAGE_SIZE + 1)
    with _client_with_mock_deps(monkeypatch) as c:
        resp = c.post(
            "/api/v1/upload?user_id=1",
            files={"file": ("f.jpg", big, "image/jpeg")},
        )
    assert resp.status_code == 413