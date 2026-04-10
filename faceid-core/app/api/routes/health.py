from typing import Any
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
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
    """
    return {"status": "ok"}


@router.get("/ready")
async def readiness(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    """
    Readiness probe — проверяет зависимости.
    Возвращает 200 только если и БД, и Redis доступны.
    """
    status = "ok"
    checks: dict[str, str] = {}

    # DB
    try:
        await db.execute(text("SELECT 1"))
        checks["db"] = "ok"
    except Exception as e:
        logger.exception("readiness_db_failed error=%s", e)
        checks["db"] = "fail"
        status = "degraded"

    # Redis
    try:
        redis_client.ping()
        checks["redis"] = "ok"
    except Exception as e:
        logger.exception("readiness_redis_failed error=%s", e)
        checks["redis"] = "fail"
        status = "degraded"

    payload: dict[str, Any] = {
        "status": status,
        "checks": checks,
    }

    if status != "ok":
        return JSONResponse(status_code=503, content=payload)

    return JSONResponse(status_code=200, content=payload)
