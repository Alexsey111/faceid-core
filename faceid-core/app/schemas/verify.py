# verify.py - Схемы для верификации

from pydantic import BaseModel
from typing import Optional, Union


class VerifyRequest(BaseModel):
    user_id: Optional[str] = None
    image: str  # base64 encoded image
    require_liveness: bool = False


class VerifyResponse(BaseModel):

    status: str
    user_id: Optional[Union[str, int]] = None
    similarity: Optional[float] = None
    liveness_passed: Optional[bool] = None