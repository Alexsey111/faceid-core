from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict


class VerifyRequest(BaseModel):
    user_id: Optional[str] = None
    image: str
    require_liveness: bool = False


class VerifyResponse(BaseModel):
    status: str
    user_id: Optional[Union[str, int]] = None
    similarity: Optional[float] = None
    liveness_passed: Optional[bool] = None
    queue_wait_ms: Optional[float] = None
    error_code: Optional[str] = None

    # Diagnostics for the quality gate.
    reason: Optional[str] = None
    quality_details: Optional[dict[str, Any]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "match",
                    "similarity": 0.87,
                    "liveness_passed": True,
                },
                {
                    "status": "spoof_detected",
                    "liveness_passed": False,
                },
                {
                    "status": "quality_reject",
                    "reason": "image_blurry",
                    "quality_details": {
                        "blur_score": 18.4,
                        "brightness": 92.1,
                        "contrast": 21.7,
                    },
                },
                {
                    "status": "processing_failed",
                    "error_code": "invalid_image",
                },
            ]
        }
    )


class VerifyEnqueueResponse(BaseModel):
    job_id: str
    status: str = "pending"
