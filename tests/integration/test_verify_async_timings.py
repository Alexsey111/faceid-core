from __future__ import annotations

import base64
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import job_status as job_status_route
from app.api.routes import verify_async as verify_async_route
from app.services import verify_result_store
from app.services.verify_result_store import VerifyResultStore


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.values[key] = value

    def get(self, key: str) -> str | None:
        return self.values.get(key)


def _make_image_b64() -> str:
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def test_verify_async_completion_includes_timings_and_timestamps(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(verify_result_store, "redis_client", fake_redis)

    # Admission раньше читал длину очереди через модульный redis_client.llen;
    # теперь — VerifyJobQueue.evaluate_admission() (возвращает AdmissionDecision).
    # Мокаем принять заявку без обращения к реальному Redis.
    async def _fake_evaluate_admission():
        return SimpleNamespace(
            accepted=True,
            reason="accepted",
            queue_len=0,
            inflight=0,
            estimated_delay_ms=0.0,
        )
    monkeypatch.setattr(
        verify_async_route.VerifyJobQueue, "evaluate_admission", _fake_evaluate_admission
    )

    # MinIO mock: route загружает фото в MinIO перед enqueue (п.4 — plaintext
    # base64 не кладётся в Redis-очередь). Реального MinIO в тесте нет.
    class _FakeMinio:
        def __init__(self):
            pass

        def upload_image(self, object_name, data, content_type="image/jpeg"):
            pass

        def delete_image(self, object_name):
            pass

    monkeypatch.setattr(verify_async_route, "MinioClient", _FakeMinio)

    async def fake_enqueue_job(payload: dict[str, object], admission=None) -> str:
        job_id = "job-test-1"
        VerifyResultStore.set_done(
            job_id,
            {
                "status": "match",
                "user_id": "42",
                "similarity": 0.99,
                "margin": 0.12,
                "timings": {
                    "queue_wait_ms": 1.0,
                    "batch_wait_ms": 2.0,
                    "worker_semaphore_wait_ms": 3.0,
                    "pipeline_total_ms": 4.0,
                    "result_write_ms": 5.0,
                    "job_total_server_ms": 6.0,
                },
                "timestamps": {
                    "accepted_at_ns": 10,
                    "enqueued_at_ns": 20,
                    "worker_claimed_at_ns": 30,
                    "processing_started_at_ns": 40,
                    "processing_finished_at_ns": 50,
                    "result_written_at_ns": 60,
                },
            },
            metrics={"created_at": 1.0, "started_at": 2.0, "finished_at": 3.0},
        )
        return job_id

    monkeypatch.setattr(verify_async_route.VerifyJobQueue, "enqueue_job", fake_enqueue_job)

    app = FastAPI()
    app.include_router(verify_async_route.router)
    app.include_router(job_status_route.router)

    client = TestClient(app)
    # Роутеры смонтированы без /api/v1 prefix (тест зеркалит только admission-
    # тайминги, не production-монтаж). GET /jobs/{id}/wait — тоже без prefix.
    response = client.post(
        "/verify_async",
        json={
            "image_b64": _make_image_b64(),
            "user_id": "42",
            "require_liveness": False,
        },
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    wait_response = client.get(f"/jobs/{job_id}/wait", params={"timeout": 100})
    assert wait_response.status_code == 200
    data = wait_response.json()

    assert data["status"] == "done"
    assert "timings" in data
    assert "timestamps" in data
    for key in (
        "accepted_at_ns",
        "enqueued_at_ns",
        "worker_claimed_at_ns",
        "processing_started_at_ns",
        "processing_finished_at_ns",
        "result_written_at_ns",
    ):
        assert key in data["timestamps"]
    assert data["timings"]["job_total_server_ms"] == 6.0
