# router.py - Основной роутер API

from fastapi import APIRouter, Depends

from app.api.deps import require_auth
from app.api.routes.verify import router as verify_router
from app.api.routes.verify_async import router as verify_async_router
from app.api.routes.upload import router as upload_router
from app.api.routes.liveness import router as liveness_router
from app.api.routes.liveness_challenge import router as liveness_challenge_router
from app.api.routes.job_status import router as job_status_router
from app.api.routes.status import router as status_router
from app.api.routes.update_reference import router as update_router

# Защита эндпоинтов (ТЗ 3.1): require_auth проверяет JWT/X-API-Key,
# коротко замыкается при AUTH_ENABLED=False (testing/dev).
# Health/ready/docs остаются открытыми (см. app/main.py — health_router без deps).
_AUTH = [Depends(require_auth)]


router = APIRouter()

router.include_router(upload_router, dependencies=_AUTH)
router.include_router(verify_router, dependencies=_AUTH)
router.include_router(verify_async_router, dependencies=_AUTH)
router.include_router(job_status_router, dependencies=_AUTH)
router.include_router(liveness_router, dependencies=_AUTH)
# liveness_challenge: /init защищён Depends(require_auth) в сигнатуре,
# /stream (WS) валидирует ws_token (выдан authed-эндпоинтом /init).
# Router-level _AUTH (HTTPBearer) к WebSocket не применяется — не вешаем.
router.include_router(liveness_challenge_router)
router.include_router(status_router, dependencies=_AUTH)
router.include_router(update_router, dependencies=_AUTH)
