# main.py - Точка входа в приложение

from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import cv2
import numpy as np
import sqlalchemy as sa
from app.core.logger import setup_logging
from app.api.router import router
from app.api.routes.health import router as health_router
from app.db.base import Base
from app.db.session import engine, AsyncSessionLocal
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.core.config import settings
from app.ml.pipeline import FacePipeline
from app.ml.dependencies import reset_batch_encoder
from app.services.faiss_index import FaissIndex
from app.services.search_service import SearchService
from app.models.user import User  # noqa: F401
from app.models.embedding import Embedding  # noqa: F401
from app.models.verification_log import VerificationLog  # noqa: F401
from app.models.verification_job import VerificationJob  # noqa: F401
from app.core.middleware import request_id_middleware

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
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

    # Initialize ML pipeline
    cv2.setNumThreads(2)
    pipeline = FacePipeline()

    # warm-up (очень важно)
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
    app.include_router(router)
    app.include_router(health_router)

    return app


app = create_app()


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
