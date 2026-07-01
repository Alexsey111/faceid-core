# app/api/routes/status.py  - Роут проверки статуса

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.db.session import get_db
from app.ml.runtime import get_face_app
from app.monitoring.db_metrics import timed_db_call

router = APIRouter()


@router.get("/status")
async def system_status(db: AsyncSession = Depends(get_db)):
    """
    System health check endpoint.
    """

    status = {
        "api": "ok",
        "database": "unknown",
        "ml_runtime": "unknown"
    }

    # Check database
    try:
        await timed_db_call(db.execute(text("SELECT 1")), "status.check")
        status["database"] = "ok"
    except Exception:
        status["database"] = "error"

    # Check ML runtime
    try:
        app = get_face_app()
        if app:
            status["ml_runtime"] = "ok"
    except Exception:
        status["ml_runtime"] = "error"

    return status
