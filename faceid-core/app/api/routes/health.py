# faceid-core\app\api\routes\health.py

from typing import Any
import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.db.session import get_db
from app.infrastructure.redis_client import redis_client

router = APIRouter()
logger = logging.getLogger("health")


@router.get("/health")
async def health() -> dict[str, str]:
    """
    Liveness probe — быстрый, без зависимостей.
    Используется для Kubernetes livenessProbe.
    """
    return {
        "status": "ok"
    }


@router.get("/ready")
async def readiness(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """
    Readiness probe — проверяет зависимости.
    Используется для Kubernetes readinessProbe.
    """

    status = "ok"
    checks: dict[str, str] = {}

    # --- DB check ---
    try:
        await db.execute(text("SELECT 1"))
        checks["db"] = "ok"
    except Exception as e:
        logger.exception("readiness_db_failed error=%s", e)
        checks["db"] = "fail"
        status = "degraded"

    # --- Redis check ---
    try:
        redis_client.ping()
        checks["redis"] = "ok"
    except Exception as e:
        logger.exception("readiness_redis_failed error=%s", e)
        checks["redis"] = "fail"
        status = "degraded"

    return {
        "status": status,
        "checks": checks
    }