# main.py - Точка входа в приложение

from contextlib import asynccontextmanager
import os
import logging
from fastapi import Depends, FastAPI, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, REGISTRY, generate_latest, multiprocess
import cv2
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
from app.models.user import User  # noqa: F401
from app.models.embedding import Embedding  # noqa: F401
from app.models.verification_log import VerificationLog  # noqa: F401
from app.models.verification_job import VerificationJob  # noqa: F401
from app.core.middleware import request_id_middleware
from app.monitoring.http_metrics import metrics_middleware

setup_logging()

logger = logging.getLogger(__name__)


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
    # Демо-GUI для презентаций (dev; AUTH_ENABLED=false в env). Same-origin:
    # статика на /demo/, API на /api/v1 — CORS не нужен. Production убирает mount
    # или ставит перед ним auth+HTTPS (см. docs/demo-guide.md).
    # Путь разрешаем от расположения app/main.py (корень репо → demo/), а не от CWD:
    # так mount работает и из Docker (CWD=/app), и при host-run, и в тестах.
    # Graceful skip при отсутствии demo/ — чтобы запуск без демо-артефактов не падал.
    _demo_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo")
    if os.path.isdir(_demo_dir):
        app.mount("/demo", StaticFiles(directory=_demo_dir, html=True), name="demo")
    # Все бизнес-эндпоинты под /api/v1 (ТЗ 4): /api/v1/verify, /api/v1/liveness,
    # /api/v1/upload и т.д. Health/ready/docs остаются без префикса (оркестратор).
    app.include_router(router, prefix="/api/v1")
    app.include_router(health_router)

    return app


app = create_app()


@app.get("/metrics", dependencies=[Depends(require_auth)])
def metrics():
    return _prometheus_metrics_response()
