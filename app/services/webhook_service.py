# app/services/webhook_service.py — Webhook-уведомления внешних систем (ТЗ 3.2).
#
# Fire-and-forget доставка уведомления о завершении верификации на внешний URL.
# Подпись тела HMAC-SHA256 (заголовок X-FaceID-Signature), идемпотентность через
# Redis SET NX webhook:sent:{job_id}. Не блокирует основной поток верификации.
#
# Два входа:
#   - notify(...)        — async, ставит задачу в bounded-очередь (воркеры с event-loop).
#   - notify_sync(...)   — sync-обёртка для синхронных callers (Celery-таски, sync-роуты):
#     если loop запущен — планирует create_task(notify); иначе — прямая доставка
#     через asyncio.run (без очереди).

import asyncio
import hashlib
import hmac
import json
import logging
from typing import Any

import httpx

from app.core.config import settings
from app.infrastructure.redis_client import redis_client
from app.monitoring.metrics import (
    WEBHOOK_DELIVERY_FAILED,
    WEBHOOK_DELIVERY_LATENCY,
    WEBHOOK_DELIVERY_TOTAL,
    WEBHOOK_QUEUE_DEPTH,
)

logger = logging.getLogger("webhook")


def _build_headers(job_id: str, body: bytes) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "X-FaceID-JobId": job_id,
    }
    secret = settings.WEBHOOK_SECRET
    if secret:
        sig = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        headers["X-FaceID-Signature"] = f"sha256={sig}"
    return headers


def _idempotency_acquire(job_id: str, state: str) -> bool:
    """
    Cross-process идемпотентность: SET NX webhook:sent:{job_id}.
    Возвращает True, если мы «захватили» право на отправку (ключа не было).
    При ошибке Redis — fail-open (возвращаем True): лучше отправить дубль,
    чем потерять уведомление.
    """
    key = f"webhook:sent:{job_id}"
    try:
        return bool(
            redis_client.set_if_absent(
                key, state, ttl=settings.WEBHOOK_IDEMPOTENCY_TTL_S
            )
        )
    except Exception:
        logger.warning("webhook_idempotency_redis_failed job_id=%s", job_id, exc_info=True)
        return True


class WebhookService:
    """Singleton с фоновой таской доставки из bounded-очереди (для async callers)."""

    _instance: "WebhookService | None" = None

    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict[str, Any]] | None = None
        self._worker_task: asyncio.Task[None] | None = None

    @classmethod
    async def get_instance(cls) -> "WebhookService":
        if cls._instance is None or cls._instance._queue is None:
            cls._instance = cls()
            await cls._instance._start()
        return cls._instance

    async def _start(self) -> None:
        if not settings.WEBHOOK_ENABLED or self._queue is not None:
            return
        self._queue = asyncio.Queue(maxsize=max(1, settings.WEBHOOK_QUEUE_SIZE))
        self._worker_task = asyncio.create_task(self._run(), name="webhook-worker")
        logger.info("webhook_service_started url=%s", settings.WEBHOOK_URL)

    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            item = await self._queue.get()
            try:
                await _deliver_with_retries(item)
            except Exception:
                logger.exception("webhook_worker_unexpected_error job_id=%s", item.get("job_id"))
            finally:
                self._queue.task_done()
                WEBHOOK_QUEUE_DEPTH.set(self._queue.qsize())

    async def notify(self, job_id: str, state: str, payload: dict[str, Any]) -> None:
        """Async fire-and-forget: идемпотентность + постановка в очередь."""
        if not settings.WEBHOOK_ENABLED:
            return
        if not settings.WEBHOOK_URL:
            logger.warning("webhook_enabled_but_no_url job_id=%s", job_id)
            return
        if not _idempotency_acquire(job_id, state):
            WEBHOOK_DELIVERY_FAILED.labels(reason="idempotent_skip").inc()
            WEBHOOK_DELIVERY_TOTAL.labels(state=state, status="dropped").inc()
            return
        item = {"job_id": job_id, "state": state, "payload": payload}
        svc = await WebhookService.get_instance()
        queue = svc._queue
        assert queue is not None
        try:
            queue.put_nowait(item)
            WEBHOOK_QUEUE_DEPTH.set(queue.qsize())
        except asyncio.QueueFull:
            WEBHOOK_DELIVERY_FAILED.labels(reason="queue_full").inc()
            WEBHOOK_DELIVERY_TOTAL.labels(state=state, status="dropped").inc()
            logger.warning("webhook_queue_full job_id=%s", job_id)

    async def shutdown(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass


async def _deliver_with_retries(item: dict[str, Any]) -> None:
    """Одна доставка с экспоненциальным backoff (общая для очереди и sync-пути)."""
    job_id = item["job_id"]
    state = item["state"]
    body = json.dumps(item, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    headers = _build_headers(job_id, body)
    max_retries = max(1, settings.WEBHOOK_MAX_RETRIES)
    url = settings.WEBHOOK_URL
    timeout = settings.WEBHOOK_TIMEOUT_S

    last_reason = "unknown"
    for attempt in range(1, max_retries + 1):
        try:
            import time as _time
            t0 = _time.perf_counter()
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, content=body, headers=headers)
            WEBHOOK_DELIVERY_LATENCY.observe((_time.perf_counter() - t0) * 1000.0)

            if 200 <= resp.status_code < 300:
                WEBHOOK_DELIVERY_TOTAL.labels(state=state, status="success").inc()
                return

            last_reason = "non_2xx"
            WEBHOOK_DELIVERY_FAILED.labels(reason="non_2xx").inc()
            logger.warning(
                "webhook_non_2xx job_id=%s status=%s body=%s",
                job_id, resp.status_code, resp.text[:200],
            )
        except httpx.TimeoutException:
            last_reason = "timeout"
            WEBHOOK_DELIVERY_FAILED.labels(reason="timeout").inc()
        except httpx.RequestError as exc:
            last_reason = "receiver_unavailable"
            WEBHOOK_DELIVERY_FAILED.labels(reason="receiver_unavailable").inc()
            logger.warning("webhook_request_error job_id=%s attempt=%s: %s", job_id, attempt, exc)

        if attempt < max_retries:
            WEBHOOK_DELIVERY_TOTAL.labels(state=state, status="retry").inc()
            await asyncio.sleep(2 ** (attempt - 1))
            continue

    WEBHOOK_DELIVERY_TOTAL.labels(state=state, status="failed").inc()
    logger.error("webhook_failed job_id=%s reason=%s after=%s", job_id, last_reason, max_retries)


def _is_enabled_and_configured() -> bool:
    if not settings.WEBHOOK_ENABLED:
        return False
    if not settings.WEBHOOK_URL:
        logger.warning("webhook_enabled_but_no_url")
        return False
    return True


async def notify_direct(job_id: str, state: str, payload: dict[str, Any]) -> None:
    """
    Прямая доставка с идемпотентностью, без очереди — для короткоживущих
    event-loops (Celery-таски, запускаемые через run_worker_coroutine/asyncio.run,
    где фоновой таске не успеть выполниться до завершения loop).
    """
    if not _is_enabled_and_configured():
        return
    if not _idempotency_acquire(job_id, state):
        WEBHOOK_DELIVERY_FAILED.labels(reason="idempotent_skip").inc()
        WEBHOOK_DELIVERY_TOTAL.labels(state=state, status="dropped").inc()
        return
    item = {"job_id": job_id, "state": state, "payload": payload}
    await _deliver_with_retries(item)


def notify_sync(job_id: str, state: str, payload: dict[str, Any]) -> None:
    """
    Sync-вход для синхронных callers (sync-роуты verify.py).
    Если event-loop запущен — планирует create_task(...) (fire-and-forget через очередь).
    Иначе — прямая доставка через asyncio.run(notify_direct(...)) (блокирующая,
    ограниченная timeout×retries). Идемпотентность проверяется внутри.
    """
    if not _is_enabled_and_configured():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Нет запущенного loop (sync caller) — доставка в новом loop.
        try:
            asyncio.run(notify_direct(job_id, state, payload))
        except Exception:
            logger.exception("webhook_sync_delivery_failed job_id=%s", job_id)
        return

    # Loop запущен — fire-and-forget через очередь.
    loop.create_task(_notify_async(job_id, state, payload))


async def _notify_async(job_id: str, state: str, payload: dict[str, Any]) -> None:
    try:
        svc = await WebhookService.get_instance()
        await svc.notify(job_id, state, payload)
    except Exception:
        logger.exception("webhook_async_dispatch_failed job_id=%s", job_id)


def fire_sync_webhook(result: Any) -> None:
    """Webhook для sync-верификации (ТЗ 3.2): синтетический job_id sync-<uuid>,
    payload sanitised через VerifyResultStore._sanitize_mapping (выкинет
    embedding/image и пр.). Fire-and-forget. Guard WEBHOOK_ENABLED — внутри
    notify_sync. Единый источник sanitize для sync-роутов verify.py — раньше
    логика дублировалась, что рискованно для sanitize (биометрия в webhook =
    нарушение 152-ФЗ при рассинхроне)."""
    try:
        from uuid import uuid4
        from app.services.verify_result_store import VerifyResultStore
        if hasattr(result, "model_dump"):
            data = result.model_dump()
        elif isinstance(result, dict):
            data = result
        else:
            return
        payload = VerifyResultStore._sanitize_mapping(data)
        notify_sync(f"sync-{uuid4()}", "sync", payload)
    except Exception:
        logger.warning("webhook_dispatch_failed (sync)", exc_info=True)