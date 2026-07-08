# upload.py - Схемы для загрузки

from pydantic import BaseModel


class UploadRequest(BaseModel):
    user_id: str
    image: str  # base64 encoded image