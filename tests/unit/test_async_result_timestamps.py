from __future__ import annotations

import json

from app.services import verify_result_store
from app.services.verify_result_store import VerifyResultStore


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.values[key] = value

    def get(self, key: str) -> str | None:
        return self.values.get(key)


def test_verify_result_store_persists_timestamps_and_strips_heavy_payload(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(verify_result_store, "redis_client", fake_redis)

    payload = {
        "status": "ok",
        "match": True,
        "embedding": [1.0, 2.0, 3.0],
        "image_b64": "should_not_persist",
        "timings": {
            "queue_wait_ms": 1.25,
            "result_write_ms": 0.75,
        },
        "timestamps": {
            "accepted_at_ns": 10,
            "enqueued_at_ns": 20,
            "worker_claimed_at_ns": 30,
            "processing_started_at_ns": 40,
            "processing_finished_at_ns": 50,
            "result_written_at_ns": 60,
        },
    }

    VerifyResultStore.set_done("job-1", payload, metrics={"created_at": 1.0, "started_at": 2.0, "finished_at": 3.0})

    stored = json.loads(fake_redis.values["job:job-1"])

    assert stored["status"] == "done"
    assert "embedding" not in stored["result"]
    assert "image_b64" not in stored["result"]
    assert stored["timings"] == {
        "queue_wait_ms": 1.25,
        "result_write_ms": 0.75,
    }
    assert stored["timestamps"] == {
        "accepted_at_ns": 10,
        "enqueued_at_ns": 20,
        "worker_claimed_at_ns": 30,
        "processing_started_at_ns": 40,
        "processing_finished_at_ns": 50,
        "result_written_at_ns": 60,
    }
