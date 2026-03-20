# router.py - Основной роутер API

from fastapi import APIRouter

from app.api.routes.verify import router as verify_router
from app.api.routes.upload import router as upload_router
from app.api.routes.liveness import router as liveness_router
from app.api.routes.status import router as status_router
from app.api.routes.update_reference import router as update_router
from app.api.routes.metrics import router as metrics_router


router = APIRouter()

router.include_router(upload_router)
router.include_router(verify_router)
router.include_router(liveness_router)
router.include_router(status_router)
router.include_router(update_router)
router.include_router(metrics_router)
