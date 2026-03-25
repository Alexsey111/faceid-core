from typing import Optional, Union

from pydantic import BaseModel, ConfigDict


class VerifyRequest(BaseModel):
    user_id: Optional[str] = None
    image: str  # base64 encoded image
    require_liveness: bool = False


class VerifyResponse(BaseModel):
    status: str
    user_id: Optional[Union[str, int]] = None
    similarity: Optional[float] = None
    liveness_passed: Optional[bool] = None
    queue_wait_ms: Optional[float] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "match",
                    "similarity": 0.87,
                    "liveness_passed": True,
                },
                {
                    "status": "spoof",
                    "liveness_passed": False,
                },
            ]
        }
    )
