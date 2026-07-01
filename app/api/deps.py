# app/api/deps.py — общие зависимости эндпоинтов.

"""
Аутентификация эндпоинтов (ТЗ 3.1).

Поддерживаются два механизма (проверяются в порядке):
  1. X-API-Key  — service-to-service (статические ключи из settings.API_KEYS).
  2. Bearer JWT — клиентские вызовы (HS256, проверяется exp; опц. issuer/audience).

Переключатель settings.AUTH_ENABLED:
  - False → зависимость коротко замыкается (testing/dev без аутентификации).
  - True  → отсутствие/невалидность учётных данных → 401.

Зависимость вешается на защищённые роутеры через dependencies=[Depends(require_auth)]
в app/api/router.py. Health/ready/openapi/docs остаются открытыми.
"""

import logging
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

logger = logging.getLogger("auth")

_bearer = HTTPBearer(auto_error=False)


async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, Any]:
    # testing/dev short-circuit — не ломает существующие тесты (conftest ставит
    # AUTH_ENABLED=false).
    if not settings.AUTH_ENABLED:
        return {"kind": "disabled"}

    # 1) X-API-Key (service-to-service).
    api_key = request.headers.get("X-API-Key")
    if api_key:
        if api_key in settings.api_keys_set:
            return {"kind": "api_key"}
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid X-API-Key",
        )

    # 2) Bearer JWT.
    if credentials is None or credentials.credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.JWT_ALG],
            issuer=settings.JWT_ISSUER or None,
            audience=settings.JWT_AUDIENCE or None,
            options={"require": ["exp"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="token expired")
    except jwt.PyJWTError as exc:
        logger.warning("jwt decode failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {"kind": "jwt", "sub": payload.get("sub")}