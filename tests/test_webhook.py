# tests/test_webhook.py — P0.5: webhook-уведомления (ТЗ 3.2).
#
# Юнит-тесты: httpx подменяется FakeClient, redis (идемпотентность) — FakeRedis.
# Без инфры (маркер `unit` → conftest пропускает миграции/redis-flush).

import asyncio
import hashlib
import hmac
import json

import pytest

import app.services.webhook_service as ws
from app.core.config import settings
from app.services.webhook_service import WebhookService, notify_direct, notify_sync

pytestmark = pytest.mark.unit


# ------------------------- Fakes -------------------------------------------

class FakeResponse:
    def __init__(self, status_code: int = 200, text: str = "ok"):
        self.status_code = status_code
        self.text = text


class FakeRedis:
    """Идемпотентность без реального Redis: SET NX in-memory."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.set_calls = 0

    def set_if_absent(self, key: str, value: str, ttl: int = 300) -> bool:
        self.set_calls += 1
        if key in self.store:
            return False
        self.store[key] = value
        return True


class FakeClient:
    """Заглушка httpx.AsyncClient. Управляется class-level состояниями."""

    calls: list[dict] = []
    statuses: list[int] = [200]
    delay: float = 0.0
    raise_cls: type[Exception] | None = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def post(self, url, content=None, headers=None):
        if FakeClient.delay:
            await asyncio.sleep(FakeClient.delay)
        status = FakeClient.statuses[len(FakeClient.calls) % len(FakeClient.statuses)]
        FakeClient.calls.append({
            "url": url,
            "content": content,
            "headers": dict(headers or {}),
            "status": status,
        })
        if FakeClient.raise_cls is not None:
            raise FakeClient.raise_cls("injected")
        return FakeResponse(status, "ok")


@pytest.fixture
def fake_redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(ws, "redis_client", fake)
    return fake


@pytest.fixture
def fake_http(monkeypatch):
    FakeClient.calls = []
    FakeClient.statuses = [200]
    FakeClient.delay = 0.0
    FakeClient.raise_cls = None
    monkeypatch.setattr(ws.httpx, "AsyncClient", FakeClient)
    return FakeClient


@pytest.fixture
def fast_sleep(monkeypatch):
    """Убирает экспоненциальный backoff в retry-тестах (без долгих sleep)."""
    async def _noop(*args, **kwargs):
        return
    monkeypatch.setattr(ws.asyncio, "sleep", _noop)


@pytest.fixture
def wh_enabled(monkeypatch):
    monkeypatch.setattr(settings, "WEBHOOK_ENABLED", True)
    monkeypatch.setattr(settings, "WEBHOOK_URL", "https://hook.example/faceid")
    monkeypatch.setattr(settings, "WEBHOOK_SECRET", "topsecret")
    monkeypatch.setattr(settings, "WEBHOOK_TIMEOUT_S", 1.0)
    monkeypatch.setattr(settings, "WEBHOOK_MAX_RETRIES", 3)
    monkeypatch.setattr(settings, "WEBHOOK_QUEUE_SIZE", 256)
    monkeypatch.setattr(settings, "WEBHOOK_IDEMPOTENCY_TTL_S", 3600)
    # Сброс singleton между тестами.
    WebhookService._instance = None
    yield
    inst = WebhookService._instance
    if inst is not None and inst._worker_task is not None:
        inst._worker_task.cancel()
    WebhookService._instance = None


# ------------------------- Tests -------------------------------------------

async def test_disabled_no_post(monkeypatch, fake_http):
    monkeypatch.setattr(settings, "WEBHOOK_ENABLED", False)
    await notify_direct("job-1", "success", {"status": "done"})
    assert FakeClient.calls == []


async def test_valid_delivery_with_hmac(wh_enabled, fake_http, fake_redis):
    await notify_direct("job-1", "success", {"status": "done", "match_score": 0.9})
    assert len(FakeClient.calls) == 1
    call = FakeClient.calls[0]
    body = call["content"]
    payload = json.loads(body)
    assert payload["job_id"] == "job-1"
    assert payload["state"] == "success"
    assert payload["payload"]["match_score"] == 0.9

    # HMAC-SHA256 подпись тела (заголовок X-FaceID-Signature).
    expected = "sha256=" + hmac.new(b"topsecret", body, hashlib.sha256).hexdigest()
    assert call["headers"]["X-FaceID-Signature"] == expected
    assert call["headers"]["X-FaceID-JobId"] == "job-1"
    assert call["headers"]["Content-Type"] == "application/json"


async def test_retry_on_500_then_success(wh_enabled, fake_http, fake_redis, fast_sleep):
    FakeClient.statuses = [500, 200]
    await notify_direct("job-r", "error", {"status": "failed"})
    assert len(FakeClient.calls) == 2
    assert FakeClient.calls[-1]["status"] == 200


async def test_all_retries_fail_on_500(wh_enabled, fake_http, fake_redis, fast_sleep):
    FakeClient.statuses = [500, 500, 500]
    await notify_direct("job-fail", "error", {"status": "failed"})
    assert len(FakeClient.calls) == 3  # WEBHOOK_MAX_RETRIES=3


async def test_timeout_retries_then_fail(wh_enabled, fake_http, fake_redis, fast_sleep):
    FakeClient.raise_cls = ws.httpx.TimeoutException
    await notify_direct("job-to", "error", {"status": "failed"})
    assert len(FakeClient.calls) == 3  # все попытки таймаут → failed


async def test_idempotency_skips_second(wh_enabled, fake_http, fake_redis):
    await notify_direct("job-idem", "success", {"status": "done"})
    assert len(FakeClient.calls) == 1
    # Второй вызов с тем же job_id — SET NX вернёт False → пропуск.
    await notify_direct("job-idem", "success", {"status": "done"})
    assert len(FakeClient.calls) == 1
    assert fake_redis.set_calls == 2


async def test_queue_path_delivery(wh_enabled, fake_http, fake_redis):
    # Пути через bounded-очередь + фоновую таску (длинкоживущий loop API/worker).
    svc = await WebhookService.get_instance()
    await svc.notify("job-q", "success", {"status": "done"})
    # Даём фоновой таске отработать.
    await asyncio.sleep(0.1)
    assert len(FakeClient.calls) == 1
    assert FakeClient.calls[0]["headers"]["X-FaceID-JobId"] == "job-q"


async def test_queue_full_drops(wh_enabled, fake_http, fake_redis):
    # Инстанс с очередью maxsize=1 и БЕЗ фоновой таски (воркер не draining).
    svc = WebhookService()
    svc._queue = asyncio.Queue(maxsize=1)
    WebhookService._instance = svc
    await svc._queue.put({"job_id": "filler", "state": "x", "payload": {}})
    # Очередь полна → следующий notify должен быть отброшен без POST.
    await svc.notify("job-overflow", "success", {"status": "done"})
    assert len(FakeClient.calls) == 0


async def test_notify_sync_with_running_loop(wh_enabled, fake_http, fake_redis):
    # Sync-вход при запущенном loop → fire-and-forget через очередь.
    notify_sync("job-sync", "sync", {"status": "ok"})
    await asyncio.sleep(0.1)
    assert len(FakeClient.calls) == 1


async def test_no_secret_no_signature_header(monkeypatch, wh_enabled, fake_http, fake_redis):
    monkeypatch.setattr(settings, "WEBHOOK_SECRET", None)
    await notify_direct("job-nosig", "success", {"status": "done"})
    assert len(FakeClient.calls) == 1
    assert "X-FaceID-Signature" not in FakeClient.calls[0]["headers"]