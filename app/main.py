# main.py - Точка входа в приложение

from contextlib import asynccontextmanager
import base64
import os
import logging
import socket
import threading
import time
from typing import Any
from fastapi import Depends, FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, REGISTRY, generate_latest, multiprocess
import cv2
import numpy as np
import sqlalchemy as sa
from app.core.logger import setup_logging
from app.api.deps import require_auth
from app.api.router import router
from app.api.routes.health import router as health_router
from app.db.base import Base
from app.db.session import engine, AsyncSessionLocal
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.core.config import settings
from app.services.faiss_index import FaissIndex
from app.services.search_service import SearchService
from app.services.verification_service_factory import get_pipeline
from app.models.user import User  # noqa: F401
from app.models.embedding import Embedding  # noqa: F401
from app.models.verification_log import VerificationLog  # noqa: F401
from app.models.verification_job import VerificationJob  # noqa: F401
from app.core.middleware import request_id_middleware
from app.monitoring.http_metrics import metrics_middleware
from app.schemas.verify import VerifyRequest

setup_logging()

logger = logging.getLogger(__name__)


FAST_WORKER_SEMAPHORE = threading.Semaphore(max(1, int(settings.FAST_WORKER_MAX_CONCURRENCY)))


def _log_service_runtime_snapshot() -> None:
    logger.info(
        "service_runtime_snapshot",
        extra=settings.service_runtime_snapshot("api"),
    )


def _prometheus_metrics_response() -> Response:
    multiproc_dir = os.getenv("PROMETHEUS_MULTIPROC_DIR")

    if multiproc_dir:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        data = generate_latest(registry)
    else:
        data = generate_latest(REGISTRY)

    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


def _jsonable_pipeline_result(value) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable_pipeline_result(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable_pipeline_result(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable_pipeline_result(item) for item in value]
    return value


def _normalize_verify_sync_result(result: dict[str, Any]) -> dict[str, Any]:
    """
    Keep the worker contract stable for the API.

    The API only depends on `embedding` to decide whether the worker result is usable.
    """
    payload: dict[str, Any] = {
        "status": result.get("status", "ok"),
        "embedding": result.get("embedding"),
        "bbox": result.get("bbox"),
        "landmarks": result.get("landmarks"),
        "liveness_passed": result.get("liveness_passed"),
        "liveness_score": result.get("liveness_score"),
        "worker_hostname": socket.gethostname(),
        "worker_pid": os.getpid(),
        "wait_for_slot_ms": result.get("wait_for_slot_ms"),
        "worker_total_ms": result.get("worker_total_ms"),
    }

    if payload["status"] == "spoof":
        payload["liveness_passed"] = bool(result.get("liveness_passed", False))
        payload["liveness_score"] = float(result.get("liveness_score", 0.0) or 0.0)

    return _jsonable_pipeline_result(payload)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _log_service_runtime_snapshot()

    # Startup: create pgvector extension and tables
    async with engine.begin() as conn:
        # Create pgvector extension if not exists
        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Create tables
        await conn.run_sync(Base.metadata.create_all)

        # Add columns to verification_logs if not exist
        await conn.execute(sa.text(
            "ALTER TABLE verification_logs ADD COLUMN IF NOT EXISTS margin FLOAT DEFAULT 0.0"
        ))
        await conn.execute(sa.text(
            "ALTER TABLE verification_logs ADD COLUMN IF NOT EXISTS liveness_score FLOAT"
        ))
        await conn.execute(sa.text(
            "ALTER TABLE verification_logs ADD COLUMN IF NOT EXISTS is_genuine BOOLEAN"
        ))

    if settings.FAISS_ENABLED:
        async with AsyncSessionLocal() as session:
            repo = EmbeddingRepository(session)

            index = FaissIndex()

            items = await repo.get_all_vectors()

            index.rebuild(items)

            SearchService._faiss_index = index

    cv2.setNumThreads(2)
    # Webhook: запускаем фоновую таску доставки заранее (если включён).
    if settings.WEBHOOK_ENABLED:
        try:
            from app.services.webhook_service import WebhookService
            await WebhookService.get_instance()
        except Exception:
            logger.warning("webhook_service_init_failed", exc_info=True)

    if settings.APP_ROLE == "fast_worker":
        # Warm up the ML pipeline only in the fast worker process.
        pipeline = get_pipeline()

        try:
            dummy = np.zeros((112, 112, 3), dtype=np.uint8)
            pipeline.process(dummy.tobytes())
        except Exception:
            pass

    yield
    # Shutdown: cleanup if needed


def create_app() -> FastAPI:

    app = FastAPI(
        title="FaceID Core",
        description="Face Recognition Service",
        version="1.0.0",
        lifespan=lifespan
    )

    app.middleware("http")(request_id_middleware)
    app.middleware("http")(metrics_middleware)
    app.include_router(router)
    app.include_router(health_router)

    return app


app = create_app()


@app.post("/verify_sync", dependencies=[Depends(require_auth)])
def verify_sync(request: VerifyRequest):
    pipeline = get_pipeline()
    image_bytes = base64.b64decode(request.image)

    wait_t0 = time.perf_counter()
    FAST_WORKER_SEMAPHORE.acquire()
    wait_for_slot_ms = (time.perf_counter() - wait_t0) * 1000.0

    run_t0 = time.perf_counter()
    worker_total_ms = 0.0
    try:
        result = pipeline.process(image_bytes)
        worker_total_ms = (time.perf_counter() - run_t0) * 1000.0
        result["wait_for_slot_ms"] = wait_for_slot_ms
        result["worker_total_ms"] = worker_total_ms
        normalized = _normalize_verify_sync_result(result)
        # Webhook для fast-worker sync-пути (ТЗ 3.2). Fire-and-forget;
        # notify_sync определит наличие loop и поставит задачу в очередь.
        try:
            from app.services.webhook_service import notify_sync as _webhook_notify_sync
            from app.services.verify_result_store import VerifyResultStore
            from uuid import uuid4
            _webhook_notify_sync(
                f"sync-{uuid4()}", "sync",
                VerifyResultStore._sanitize_mapping(normalized),
            )
        except Exception:
            logger.warning("webhook_dispatch_failed (verify_sync)", exc_info=True)
        return normalized
    finally:
        if worker_total_ms == 0.0:
            worker_total_ms = (time.perf_counter() - run_t0) * 1000.0
        logger.warning(
            "fast_worker wait_for_slot_ms=%.2f worker_total_ms=%.2f",
            wait_for_slot_ms,
            worker_total_ms,
        )
        FAST_WORKER_SEMAPHORE.release()


@app.get("/metrics", dependencies=[Depends(require_auth)])
def metrics():
    return _prometheus_metrics_response()
