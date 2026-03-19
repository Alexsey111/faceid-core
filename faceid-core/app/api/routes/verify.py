# app/api/routes/verify.py

from fastapi import APIRouter, UploadFile, File, Query, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import base64

from app.db.session import get_db
from app.services.rate_limiter import RateLimiter
from app.services.verification_service_factory import get_verification_service

from app.schemas.verify import VerifyResponse, VerifyRequest


router = APIRouter()

MAX_IMAGE_SIZE = 5 * 1024 * 1024


@router.post("/verify", response_model=VerifyResponse)
async def verify_file(
    http_request: Request,
    file: UploadFile = File(...),
    user_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Verify face against reference via multipart/form-data.
    """
    print("VERIFY ENDPOINT HIT (file)")
    RateLimiter.check(http_request, "verify", limit=10)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid image format"
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="Image too large"
        )

    service = get_verification_service(db)

    result = await service.verify_face(
        image_bytes,
        user_id=user_id
    )

    return result


@router.post("/verify_base64", response_model=VerifyResponse)
async def verify_base64(
    request: VerifyRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Verify face against reference via JSON with base64 image.
    """
    print("VERIFY ENDPOINT HIT (base64)")
    print("VERIFY HIT")
    RateLimiter.check(http_request, "verify", limit=10)

    image_bytes = base64.b64decode(request.image)

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="Image too large"
        )

    service = get_verification_service(db)

    result = await service.verify_face(
        image_bytes,
        user_id=request.user_id,
        require_liveness=request.require_liveness
    )

    return result
