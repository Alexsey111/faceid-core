# app/api/routes/update_reference.py - Роут обновления референсного изображения


from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.embedding_service import EmbeddingService
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.session import get_db

router = APIRouter()


@router.put("/update-reference")
async def update_reference(
    user_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Replace user's reference face embedding.
    """

    try:

        image_bytes = await file.read()

        embedding_repo = EmbeddingRepository(db)
        service = EmbeddingService(embedding_repo)

        # удалить старый embedding
        await embedding_repo.delete_by_user_id(user_id)

        # создать новый
        result = await service.enroll_face(
            user_id=user_id,
            image_bytes=image_bytes
        )

        return {
            "updated": True,
            "user_id": user_id,
            "embedding_id": result["embedding_id"]
        }

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
