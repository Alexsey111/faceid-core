import asyncio
import base64
import json
import runpy
import time
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi import HTTPException
import pytest
from starlette.requests import Request

from app.api.routes import job_status as job_status_route
from app.api.routes import verify as verify_route
from app.api.routes import verify_async as verify_async_route
from app.schemas.verify import VerifyRequest
from app.services import verify_job_queue, verify_result_store
from app.workers import verify_worker


class FakeRedis:
    def __init__(self):
        self.set_calls = []
        self.get_values = {}
        self.rpush_calls = []
        self.blpop_values = []
        self.queue_values = {}
        self.inflight = 0

    def setex(self, key, ttl, value):
        self.set_calls.append((key, ttl, value))
        self.get_values[key] = value

    def get(self, key):
        return self.get_values.get(key)

    def rpush(self, key, value):
        self.rpush_calls.append((key, value))
        self.queue_values.setdefault(key, []).append(value)

    def incr(self, key):
        if key == "inflight_jobs":
            self.inflight += 1
            self.get_values[key] = str(self.inflight)
            return self.inflight
        raise AttributeError(key)

    def decr(self, key):
        if key == "inflight_jobs":
            self.inflight = max(0, self.inflight - 1)
            self.get_values[key] = str(self.inflight)
            return self.inflight
        raise AttributeError(key)

    def eval(self, script, numkeys, key, delta):
        if key == "inflight_jobs":
            self.inflight = max(0, self.inflight - int(delta))
            self.get_values[key] = str(self.inflight)
            return self.inflight
        raise AttributeError(key)

    def llen(self, key):
        return len(self.queue_values.get(key, []))

    def lpop(self, key):
        queue = self.queue_values.get(key, [])
        if queue:
            return queue.pop(0)
        return None

    def blpop(self, keys):
        if self.blpop_values:
            value = self.blpop_values.pop(0)
            if isinstance(value, Exception):
                raise value
            return value
        for key in keys:
            queue = self.queue_values.get(key, [])
            if queue:
                return key.encode("utf-8"), queue.pop(0)
        return None

    def brpop(self, keys, timeout=0):
        return self.blpop(keys)


class DummySemaphore:
    def __init__(self):
        self.entered = 0
        self.exited = 0

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited += 1
        return False


class DummyAsyncSession:
    def __init__(self):
        self.added = []
        self.flushed = False
        self.committed = False

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        self.flushed = True

    async def commit(self):
        self.committed = True


class DummySessionManager:
    def __init__(self, db):
        self.db = db

    async def __aenter__(self):
        return self.db

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeUploadFile:
    def __init__(self, content_type="image/jpeg", data=b"image-bytes", filename="photo.jpg"):
        self.content_type = content_type
        self._data = data
        self.filename = filename

    async def read(self):
        return self._data


@pytest.mark.asyncio
async def test_verify_result_store_set_and_get(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(verify_result_store, "redis_client", fake_redis)

    verify_result_store.VerifyResultStore.set_done("job-1", {"status": "ok"}, {})
    verify_result_store.VerifyResultStore.set_error("job-2", "boom", {})

    done_key, done_ttl, done_payload = fake_redis.set_calls[0]
    error_key, error_ttl, error_payload = fake_redis.set_calls[1]

    assert done_key == "job:job-1"
    assert done_ttl == verify_result_store.VerifyResultStore.TTL
    assert json.loads(done_payload) == {"status": "done", "result": {"status": "ok"}}

    assert error_key == "job:job-2"
    assert error_ttl == verify_result_store.VerifyResultStore.TTL
    assert json.loads(error_payload) == {"status": "error", "error": "boom"}

    fake_redis.get_values["job:job-3"] = json.dumps({"status": "done", "result": {"x": 1}}).encode("utf-8")
    assert verify_result_store.VerifyResultStore.get("job-3") == {
        "status": "done",
        "result": {"x": 1},
    }
    assert verify_result_store.VerifyResultStore.get("missing") is None


def test_normalize_priority():
    assert verify_route._normalize_priority(None) == ("high", 9)
    assert verify_route._normalize_priority(" low ") == ("low", 0)
    with pytest.raises(HTTPException, match="Invalid priority"):
        verify_route._normalize_priority("medium")


@pytest.mark.asyncio
async def test_call_fast_worker_variants():
    class FakeResponse:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            return None

        def json(self):
            return self._data

    class FakeClient:
        def __init__(self, data):
            self.data = data
            self.calls = []

        async def post(self, url, json):
            self.calls.append((url, json))
            return FakeResponse(self.data)

    client = FakeClient({"status": "no_face"})
    data, elapsed = await verify_route._call_fast_worker(client, "http://worker", {"x": 1})
    assert data == {"status": "no_face"}
    assert elapsed >= 0
    assert client.calls[0][0] == "http://worker/verify_sync"

    client = FakeClient({"status": "ok", "embedding": [1, 2, 3]})
    data, _ = await verify_route._call_fast_worker(client, "http://worker", {"x": 1})
    assert data["embedding"] == [1, 2, 3]

    client = FakeClient({"status": "ok"})
    with pytest.raises(HTTPException, match="invalid payload"):
        await verify_route._call_fast_worker(client, "http://worker", {"x": 1})


@pytest.mark.asyncio
async def test_enqueue_verify_job_success_and_failure(monkeypatch):
    captured = {}

    class FakeJobRepo:
        def __init__(self, db):
            self.db = db

        async def create(self, **kwargs):
            captured["create"] = kwargs

        async def update(self, job_id, **kwargs):
            captured["update"] = (job_id, kwargs)

    class FakeMinio:
        def upload_image(self, *args, **kwargs):
            captured["upload"] = (args, kwargs)

    class FakeTask:
        def apply_async(self, **kwargs):
            captured["task"] = kwargs

    db = DummyAsyncSession()
    monkeypatch.setattr(verify_route, "VerificationJobRepository", FakeJobRepo)
    monkeypatch.setattr(verify_route, "MinioClient", FakeMinio)
    monkeypatch.setattr(verify_route, "verify_task", FakeTask())

    await verify_route._enqueue_verify_job(
        db=db,
        job_id="job-1",
        request_received_time=123.0,
        image_bytes=b"img",
        object_name="verify/job-1/legacy.jpg",
        content_type="image/jpeg",
        user_id="7",
        require_liveness=True,
        priority="high",
    )

    assert captured["create"] == {"job_id": "job-1", "status": verify_route.JobStatus.pending}
    assert captured["upload"][0] == ("verify/job-1/legacy.jpg", b"img", "image/jpeg")
    assert captured["task"]["queue"] == "verify_heavy"
    assert captured["task"]["priority"] == 9

    class BrokenMinio:
        def upload_image(self, *args, **kwargs):
            raise RuntimeError("upload failed")

    captured.clear()
    monkeypatch.setattr(verify_route, "MinioClient", BrokenMinio)
    with pytest.raises(HTTPException, match="Failed to enqueue verify job"):
        await verify_route._enqueue_verify_job(
            db=db,
            job_id="job-2",
            request_received_time=123.0,
            image_bytes=b"img",
            object_name="verify/job-2/legacy.jpg",
            content_type="image/jpeg",
            user_id=None,
            require_liveness=False,
            priority="low",
        )
    assert captured["update"][0] == "job-2"
    assert captured["update"][1]["status"] == verify_route.JobStatus.failed


@pytest.mark.asyncio
async def test_verify_job_queue_enqueue(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(verify_job_queue, "redis_client", fake_redis)
    monkeypatch.setattr(verify_job_queue.uuid, "uuid4", lambda: "job-123")
    monkeypatch.setattr(verify_job_queue.time, "time", lambda: 123.456)

    job_id = verify_job_queue.VerifyJobQueue.enqueue(
        {"image_url": "verify_async/job-123/image.jpg", "user_id": "7", "require_liveness": True}
    )

    assert job_id == "job-123"
    assert fake_redis.rpush_calls[0][0] == verify_job_queue.VerifyJobQueue.QUEUE_NAME

    queued_job = json.loads(fake_redis.rpush_calls[0][1])
    assert queued_job["job_id"] == "job-123"
    assert queued_job["payload"]["user_id"] == "7"
    assert queued_job["created_at"] == 123.456

    status_key, ttl, payload = fake_redis.set_calls[0]
    assert status_key == "job:job-123"
    assert ttl == 300
    assert json.loads(payload) == {"status": "processing"}


@pytest.mark.asyncio
async def test_verify_async_route_enqueues(monkeypatch):
    captured = {}
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

    # MinIO mock: route загружает фото в MinIO, в payload идёт image_url (не base64)
    class _FakeMinio:
        def __init__(self):
            self.uploaded = []

        def upload_image(self, object_name, data, content_type="image/jpeg"):
            self.uploaded.append((object_name, data))

        def delete_image(self, object_name):
            pass

    fake_minio = _FakeMinio()
    monkeypatch.setattr(verify_async_route, "MinioClient", lambda *a, **kw: fake_minio)

    async def fake_evaluate_admission():
        return verify_job_queue.AdmissionDecision(accepted=True, reason="accepted")

    monkeypatch.setattr(
        verify_async_route.VerifyJobQueue, "evaluate_admission",
        staticmethod(fake_evaluate_admission),
    )

    def fake_enqueue(payload, admission=None):
        captured["payload"] = payload
        return "job-xyz"

    monkeypatch.setattr(verify_async_route.VerifyJobQueue, "enqueue", staticmethod(fake_enqueue))

    body = json.dumps(
        {
            "image_b64": image_b64,
            "user_id": "42",
            "require_liveness": True,
        }
    ).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/verify_async",
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

    result = await verify_async_route.verify_async(request)

    assert result == {"job_id": "job-xyz", "status": "queued"}
    # photo загружено в MinIO
    assert len(fake_minio.uploaded) == 1
    up_name, up_bytes = fake_minio.uploaded[0]
    assert up_name.startswith("verify_async/")
    # payload: image_url есть, plaintext image_b64 отсутствует (152-ФЗ)
    payload = captured["payload"]
    assert payload["image_url"] == up_name
    assert "image_b64" not in payload
    assert payload["user_id"] == "42"
    assert payload["require_liveness"] is True
    assert "accepted_at_ns" in payload and "enqueued_at_ns" in payload


@pytest.mark.asyncio
async def test_verify_file_route_success_and_validation(monkeypatch):
    monkeypatch.setattr(verify_route.RateLimiter, "check", lambda *args, **kwargs: None)

    class FakeService:
        async def verify_face(self, image_bytes, user_id=None):
            return {
                "status": "match",
                "user_id": user_id,
                "similarity": 0.88,
                "liveness_passed": True,
            }

    monkeypatch.setattr(verify_route, "get_verification_service", lambda db: FakeService())

    result = await verify_route.verify_file(
        http_request=SimpleNamespace(),
        file=FakeUploadFile(),
        user_id="11",
        db=SimpleNamespace(),
    )
    assert result["status"] == "match"
    assert result["user_id"] == "11"

    with pytest.raises(HTTPException, match="Invalid image format"):
        await verify_route.verify_file(
            http_request=SimpleNamespace(),
            file=FakeUploadFile(content_type="text/plain"),
            user_id=None,
            db=SimpleNamespace(),
        )


@pytest.mark.asyncio
async def test_verify_async_file_route_success_and_backpressure(monkeypatch):
    captured = {}
    monkeypatch.setattr(verify_route, "try_reserve_slot", lambda **kwargs: True)
    monkeypatch.setattr(verify_route.RateLimiter, "check", lambda *args, **kwargs: None)

    async def fake_enqueue_verify_job(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(verify_route, "_enqueue_verify_job", fake_enqueue_verify_job)

    result = await verify_route.verify_async(
        http_request=SimpleNamespace(),
        file=FakeUploadFile(filename="face.png"),
        user_id="7",
        require_liveness=True,
        priority="low",
        db=SimpleNamespace(),
    )
    assert result["status"] == "pending"
    assert captured["object_name"].endswith("face.png")
    assert captured["require_liveness"] is True
    assert captured["user_id"] == "7"

    monkeypatch.setattr(verify_route, "try_reserve_slot", lambda **kwargs: False)
    with pytest.raises(HTTPException, match="Backpressure: queue_delay_sla"):
        await verify_route.verify_async(
            http_request=SimpleNamespace(),
            file=FakeUploadFile(),
            user_id=None,
            require_liveness=False,
            priority="high",
            db=SimpleNamespace(),
        )


@pytest.mark.asyncio
async def test_verify_async_base64_validation(monkeypatch):
    monkeypatch.setattr(verify_route, "try_reserve_slot", lambda **kwargs: True)
    monkeypatch.setattr(verify_route.RateLimiter, "check", lambda *args, **kwargs: None)

    async def fake_enqueue_verify_job(**kwargs):
        return None

    monkeypatch.setattr(verify_route, "_enqueue_verify_job", fake_enqueue_verify_job)

    result = await verify_route.verify_async_base64(
        request=VerifyRequest(image=base64.b64encode(b"img").decode("utf-8"), user_id=None),
        http_request=SimpleNamespace(),
        priority="high",
        db=SimpleNamespace(),
    )
    assert result["status"] == "pending"

    with pytest.raises(HTTPException, match="Invalid base64"):
        await verify_route.verify_async_base64(
            request=VerifyRequest(image="not-base64!", user_id=None),
            http_request=SimpleNamespace(),
            priority="high",
            db=SimpleNamespace(),
        )


@pytest.mark.asyncio
async def test_get_verify_result_cached_and_db_paths(monkeypatch):
    class FakeRedis:
        def __init__(self, value=None):
            self.value = value

        def get(self, key):
            return self.value

    monkeypatch.setattr(verify_route, "redis_client", FakeRedis(json.dumps({"status": "done", "result": {"ok": True}}).encode("utf-8")))
    result = await verify_route.get_verify_result("job-cached", db=SimpleNamespace())
    assert result == {"status": "done", "result": {"ok": True}}

    monkeypatch.setattr(verify_route, "redis_client", FakeRedis(None))

    class PendingJob:
        status = SimpleNamespace(value="pending")
        result = None
        error = None

    class DoneJob:
        status = "done"
        result = {"foo": "bar"}
        error = None

    class FakeRepo:
        def __init__(self, db):
            self.db = db

        async def get_by_id(self, job_id):
            if job_id == "pending":
                return PendingJob()
            if job_id == "done":
                return DoneJob()
            return None

    monkeypatch.setattr(verify_route, "VerificationJobRepository", FakeRepo)

    pending = await verify_route.get_verify_result("pending", db=SimpleNamespace())
    assert pending == {"job_id": "pending", "status": "pending", "ready": False}

    done = await verify_route.get_verify_result("done", db=SimpleNamespace())
    assert done == {
        "job_id": "done",
        "status": "done",
        "result": {"foo": "bar"},
        "error": None,
        "ready": True,
    }

    with pytest.raises(HTTPException, match="Job not found"):
        await verify_route.get_verify_result("missing", db=SimpleNamespace())


@pytest.mark.asyncio
async def test_job_status_route_not_found_and_found(monkeypatch):
    monkeypatch.setattr(job_status_route.VerifyResultStore, "get", staticmethod(lambda job_id: None))
    assert await job_status_route.get_job_status("missing") == {
        "job_id": "missing",
        "status": "not_found",
    }

    monkeypatch.setattr(
        job_status_route.VerifyResultStore,
        "get",
        staticmethod(lambda job_id: {"status": "done", "result": {"ok": True}}),
    )
    assert await job_status_route.get_job_status("job-1") == {
        "job_id": "job-1",
        "status": "done",
        "result": {"ok": True},
    }


@pytest.mark.asyncio
async def test_verify_base64_fallbacks(monkeypatch):
    captured = {}
    monkeypatch.setattr(verify_route.settings, "USE_FAST_PATH", True)
    monkeypatch.setattr(verify_route, "is_fast_worker_enabled", lambda: True)
    monkeypatch.setattr(verify_route, "should_use_async", lambda: False)
    monkeypatch.setattr(verify_route, "try_reserve_fast_path_slot", lambda: False)
    monkeypatch.setattr(verify_route, "get_fast_worker_failures", lambda: 2)
    monkeypatch.setattr(verify_route.RateLimiter, "check", lambda *args, **kwargs: None)
    monkeypatch.setattr(verify_route, "decrement_active", lambda: None)

    async def fake_enqueue_verify_job(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(verify_route, "_enqueue_verify_job", fake_enqueue_verify_job)

    result = await verify_route.verify_base64(
        VerifyRequest(
            image=base64.b64encode(b"image-bytes").decode("utf-8"),
            user_id="55",
            require_liveness=False,
        ),
        http_request=SimpleNamespace(),
        db=SimpleNamespace(),
    )

    assert result["status"] == "pending"
    assert captured["user_id"] == "55"


@pytest.mark.asyncio
async def test_verify_worker_run_worker_processes_one_job(monkeypatch):
    seen = {}
    batches = [
        [
            {
                "job_id": "job-3",
                "payload": {"image_url": "verify_async/job-3/image.jpg"},
            }
        ]
    ]

    async def fake_collect_batch():
        if batches:
            return batches.pop(0)
        raise RuntimeError("stop")

    async def fake_process_batch(job_data):
        seen["job_data"] = job_data
        raise RuntimeError("stop")

    monkeypatch.setattr(verify_worker, "collect_batch", fake_collect_batch)
    monkeypatch.setattr(verify_worker, "process_batch", fake_process_batch)
    monkeypatch.setattr(verify_worker, "start_http_server", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="stop"):
        await verify_worker.run_worker()

    assert seen["job_data"][0]["job_id"] == "job-3"
    assert seen["job_data"][0]["payload"]["image_url"]


@pytest.mark.asyncio
async def test_verify_worker_run_worker_skips_empty_queue(monkeypatch):
    calls = {"process_batch": 0}

    batches = [[]]

    async def fake_collect_batch():
        if batches:
            return batches.pop(0)
        raise RuntimeError("stop")

    async def fake_process_batch(*args, **kwargs):
        calls["process_batch"] += 1

    monkeypatch.setattr(verify_worker, "collect_batch", fake_collect_batch)
    monkeypatch.setattr(verify_worker, "process_batch", fake_process_batch)
    monkeypatch.setattr(verify_worker, "start_http_server", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="stop"):
        await verify_worker.run_worker()

    assert calls["process_batch"] == 0


@pytest.mark.asyncio
async def test_verify_worker_main_guard(monkeypatch):
    calls = {}

    def fake_asyncio_run(coro):
        calls["called"] = True
        coro.close()
        return None

    monkeypatch.setattr(asyncio, "run", fake_asyncio_run)

    runpy.run_module("app.workers.verify_worker", run_name="__main__")

    assert calls["called"] is True


@pytest.mark.asyncio
async def test_verify_base64_fast_path_uses_service_without_pipeline(monkeypatch):
    captured = {}

    monkeypatch.setattr(verify_route.settings, "USE_FAST_PATH", True)
    monkeypatch.setattr(verify_route, "is_fast_worker_enabled", lambda: True)
    monkeypatch.setattr(verify_route, "should_use_async", lambda: False)
    monkeypatch.setattr(verify_route, "try_reserve_fast_path_slot", lambda: True)
    monkeypatch.setattr(verify_route, "record_fast_worker_success", lambda: None)
    monkeypatch.setattr(verify_route, "decrement_active", lambda: None)
    monkeypatch.setattr(verify_route.RateLimiter, "check", lambda *args, **kwargs: None)
    monkeypatch.setattr(verify_route, "_call_fast_worker", lambda *args, **kwargs: asyncio.sleep(0, result=(
        {
            "embedding": [0.1, 0.2],
            "status": "ok",
        },
        12.3,
    )))

    class FakeService:
        async def verify_from_pipeline_result(self, pipeline_result, **kwargs):
            captured["pipeline_result"] = pipeline_result
            captured["kwargs"] = kwargs
            return {"status": "match", "similarity": 0.99}

    def fake_get_service_without_pipeline(db):
        captured["db"] = db
        return FakeService()

    monkeypatch.setattr(verify_route, "get_verification_service_without_pipeline", fake_get_service_without_pipeline)
    monkeypatch.setattr(verify_route, "get_verification_service", lambda db: (_ for _ in ()).throw(AssertionError("unexpected pipeline service")))

    response = await verify_route.verify_base64(
        VerifyRequest(
            image=base64.b64encode(b"image-bytes").decode("utf-8"),
            user_id="55",
            require_liveness=True,
        ),
        http_request=SimpleNamespace(),
        db=SimpleNamespace(),
    )

    assert response == {"status": "match", "similarity": 0.99}
    assert captured["kwargs"]["image_bytes"] == b"image-bytes"
    assert captured["kwargs"]["user_id"] == "55"
    assert captured["kwargs"]["require_liveness"] is True
    assert captured["pipeline_result"]["status"] == "ok"


@pytest.mark.asyncio
async def test_verify_base64_falls_back_to_async_queue(monkeypatch):
    captured = {}

    monkeypatch.setattr(verify_route.settings, "USE_FAST_PATH", False)
    monkeypatch.setattr(verify_route.RateLimiter, "check", lambda *args, **kwargs: None)
    monkeypatch.setattr(verify_route, "decrement_active", lambda: None)

    async def fake_enqueue_verify_job(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(verify_route, "_enqueue_verify_job", fake_enqueue_verify_job)

    response = await verify_route.verify_base64(
        VerifyRequest(
            image=base64.b64encode(b"image-bytes").decode("utf-8"),
            user_id="55",
            require_liveness=False,
        ),
        http_request=SimpleNamespace(),
        db=SimpleNamespace(),
    )

    assert response["status"] == "pending"
    assert "job_id" in response
    assert captured["user_id"] == "55"
    assert captured["image_bytes"] == b"image-bytes"
    assert captured["require_liveness"] is False
