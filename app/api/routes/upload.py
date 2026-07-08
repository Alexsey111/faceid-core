# app/api/routes/upload.py- Роут загрузки изображений

from fastapi import APIRouter, UploadFile, File, Query, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
import base64

from app.services.embedding_service import EmbeddingService
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.user_repo import UserRepository
from app.db.session import get_db
from app.schemas.response import SuccessResponse
from app.schemas.upload import UploadRequest
from app.services.rate_limiter import RateLimiter
from app.api._helpers import MAX_IMAGE_SIZE


router = APIRouter()


@router.post("/upload", response_model=SuccessResponse)
async def upload_file(
    http_request: Request,
    user_id: str = Query(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload reference face for a user via multipart/form-data.
    """
    RateLimiter.check(http_request, "upload", limit=5)

    # Валидация входа (consistency с /verify): MIME + размер.
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}")
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="empty image")
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="image too large (max 5MB)")

    try:
        embedding_repo = EmbeddingRepository(db)
        user_repo = UserRepository(db)
        service = EmbeddingService(embedding_repo, user_repo)

        result = await service.enroll_face(
            user_id=int(user_id),
            image_bytes=image_bytes
        )

        return SuccessResponse(
            success=True,
            data=result
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


@router.post("/upload_base64", response_model=SuccessResponse)
async def upload_base64(
    request: UploadRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Upload reference face for a user via JSON with base64 image.
    """
    RateLimiter.check(http_request, "upload", limit=5)

    try:
        image_bytes = base64.b64decode(request.image)

        # Валидация размера (consistency с /verify: MAX_IMAGE_SIZE 5MB).
        if len(image_bytes) == 0:
            raise ValueError("empty image")
        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise ValueError("image too large (max 5MB)")

        embedding_repo = EmbeddingRepository(db)
        user_repo = UserRepository(db)
        service = EmbeddingService(embedding_repo, user_repo)

        result = await service.enroll_face(
            user_id=int(request.user_id),
            image_bytes=image_bytes
        )

        return SuccessResponse(
            success=True,
            data=result
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
