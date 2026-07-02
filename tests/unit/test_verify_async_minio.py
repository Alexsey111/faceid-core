# test_verify_async_minio.py — unit-тесты MinIO-контракта /verify_async (п.4 аудита).
#
# Plaintext base64 НЕ должен попадать в Redis-очередь: route загружает фото в MinIO
# и передаёт в payload только image_url; воркер скачивает по image_url и удаляет
# объект после обработки. Полностью mock (MinioClient, evaluate_admission, enqueue)
# → маркер 'unit' пропускает alembic/flushdb.
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from starlette.requests import Request

import app.api.routes.verify_async as verify_async_route
import app.services.verify_job_queue as verify_job_queue
import app.workers.verify_worker as verify_worker
from app.services.verify_job_queue import AdmissionDecision

_unit = pytest.mark.unit


class FakeMinio:
    """Замена MinioClient: нет сетевых вызовов/_ensure_bucket."""

    def __init__(self) -> None:
        self.uploaded: list[tuple[str, bytes]] = []
        self.deleted: list[str] = []
        self._get_bytes: bytes | None = b""

    def upload_image(self, object_name: str, data: bytes, content_type: str = "image/jpeg") -> None:
        self.uploaded.append((object_name, data))

    def delete_image(self, object_name: str) -> None:
        self.deleted.append(object_name)

    def get_image(self, object_name: str) -> bytes:
        if self._get_bytes is None:
            raise RuntimeError("minio get failed")
        return self._get_bytes


def _jpg_bytes(h: int = 200, w: int = 200) -> bytes:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _jpg_b64() -> str:
    import base64
    return base64.b64encode(_jpg_bytes()).decode("ascii")


def _make_request(body: bytes) -> Request:
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/verify_async",
            "headers": [],
            "query_string": b"",
            "client": ("testclient", 12345),
            "server": ("testserver", 80),
            "scheme": "http",
            "root_path": "",
            "http_version": "1.1",
        },
        receive,
    )


# ---------------------------------------------------------------------------
# Route: MinIO upload + image_url в payload (без plaintext base64)
# ---------------------------------------------------------------------------

@_unit
async def test_route_uploads_to_minio_and_payload_has_image_url(monkeypatch):
    """Route загружает image_bytes в MinIO, payload содержит image_url, НЕ image_b64."""
    fake_minio = FakeMinio()
    monkeypatch.setattr(verify_async_route, "MinioClient", lambda *a, **kw: fake_minio)

    async def fake_evaluate_admission():
        return AdmissionDecision(accepted=True, reason="accepted")

    monkeypatch.setattr(
        verify_async_route.VerifyJobQueue, "evaluate_admission",
        staticmethod(fake_evaluate_admission),
    )

    captured: dict = {}

    def fake_enqueue(payload, admission=None):
        captured["payload"] = payload
        return "job-1"

    monkeypatch.setattr(verify_async_route.VerifyJobQueue, "enqueue", staticmethod(fake_enqueue))

    jpg = _jpg_bytes()
    import base64 as _b64
    body = json.dumps({"image_b64": _b64.b64encode(jpg).decode("ascii"), "user_id": "42"}).encode()
    result = await verify_async_route.verify_async(_make_request(body))

    assert result == {"job_id": "job-1", "status": "queued"}
    assert len(fake_minio.uploaded) == 1
    up_name, up_bytes = fake_minio.uploaded[0]
    assert up_bytes == jpg
    assert up_name.startswith("verify_async/")
    assert captured["payload"]["image_url"] == up_name
    assert "image_b64" not in captured["payload"]
    assert captured["payload"]["user_id"] == "42"


@_unit
async def test_route_cleans_up_minio_object_on_enqueue_failure(monkeypatch):
    """Enqueue упал → загруженный объект удалён (anti-leak)."""
    fake_minio = FakeMinio()
    monkeypatch.setattr(verify_async_route, "MinioClient", lambda *a, **kw: fake_minio)

    async def fake_evaluate_admission():
        return AdmissionDecision(accepted=True, reason="accepted")

    monkeypatch.setattr(
        verify_async_route.VerifyJobQueue, "evaluate_admission",
        staticmethod(fake_evaluate_admission),
    )

    def fake_enqueue(payload, admission=None):
        raise RuntimeError("queue broken")

    monkeypatch.setattr(verify_async_route.VerifyJobQueue, "enqueue", staticmethod(fake_enqueue))

    import base64 as _b64
    body = json.dumps({"image_b64": _b64.b64encode(_jpg_bytes()).decode("ascii")}).encode()

    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        await verify_async_route.verify_async(_make_request(body))
    assert exc.value.status_code == 503
    assert len(fake_minio.uploaded) == 1
    assert fake_minio.deleted == [fake_minio.uploaded[0][0]], "объект удалён после ошибки enqueue"


@_unit
async def test_route_minio_upload_failure_returns_503(monkeypatch):
    """MinIO upload упал → 503, enqueue не вызывается."""

    def _ctor(*a, **kw):
        m = FakeMinio()
        m.upload_image = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("minio down"))
        return m

    monkeypatch.setattr(verify_async_route, "MinioClient", _ctor)

    async def fake_evaluate_admission():
        return AdmissionDecision(accepted=True, reason="accepted")

    monkeypatch.setattr(
        verify_async_route.VerifyJobQueue, "evaluate_admission",
        staticmethod(fake_evaluate_admission),
    )

    enqueue_called = {"v": False}

    def fake_enqueue(payload, admission=None):
        enqueue_called["v"] = True
        return "x"

    monkeypatch.setattr(verify_async_route.VerifyJobQueue, "enqueue", staticmethod(fake_enqueue))

    import base64 as _b64
    body = json.dumps({"image_b64": _b64.b64encode(_jpg_bytes()).decode("ascii")}).encode()

    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        await verify_async_route.verify_async(_make_request(body))
    assert exc.value.status_code == 503
    assert enqueue_called["v"] is False


# ---------------------------------------------------------------------------
# Worker: fetch image_url из MinIO + cleanup
# ---------------------------------------------------------------------------

@_unit
def test_worker_decode_job_payload_fetches_from_minio(monkeypatch):
    """_decode_job_payload_sync скачивает по image_url через MinioClient.get_image."""
    jpg = _jpg_bytes()
    fake_minio = FakeMinio()
    fake_minio._get_bytes = jpg
    monkeypatch.setattr(verify_worker, "MinioClient", lambda *a, **kw: fake_minio)

    image_bytes, image, timings = verify_worker._decode_job_payload_sync("verify_async/abc/image.jpg")
    assert image_bytes == jpg
    assert image is not None and image.shape[:2] == (200, 200)
    assert "minio_download_ms" in timings
    assert "b64_decode_ms" not in timings  # base64-путь больше не используется


@_unit
def test_worker_cleanup_minio_image_calls_delete(monkeypatch):
    fake_minio = FakeMinio()
    monkeypatch.setattr(verify_worker, "MinioClient", lambda *a, **kw: fake_minio)

    verify_worker._cleanup_minio_image("verify_async/x/i.jpg", "job-1", stage="test")
    assert fake_minio.deleted == ["verify_async/x/i.jpg"]


@_unit
def test_worker_cleanup_minio_image_noop_on_none(monkeypatch):
    called = {"delete": False}
    fake_minio = FakeMinio()
    fake_minio.delete_image = lambda name: called.__setitem__("delete", True)
    monkeypatch.setattr(verify_worker, "MinioClient", lambda *a, **kw: fake_minio)

    verify_worker._cleanup_minio_image(None, "job-1")
    verify_worker._cleanup_minio_image("", "job-1")
    assert called["delete"] is False, "None/empty image_url → delete не вызывается"


@_unit
def test_worker_cleanup_minio_image_swallows_delete_error(monkeypatch):
    """Ошибка удаления не пробрасывается (MinIO lifecycle-cover)."""
    def _ctor(*a, **kw):
        m = FakeMinio()
        m.delete_image = lambda name: (_ for _ in ()).throw(RuntimeError("minio gone"))
        return m

    monkeypatch.setattr(verify_worker, "MinioClient", _ctor)
    # не должно бросать
    verify_worker._cleanup_minio_image("verify_async/x/i.jpg", "job-1")