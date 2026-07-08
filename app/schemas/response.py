# response.py - Общие схемы ответа

from pydantic import BaseModel
from typing import Any, Optional


class SuccessResponse(BaseModel):
    """Standard success response."""
    success: bool = True
    data: Optional[Any] = None
    message: Optional[str] = None
