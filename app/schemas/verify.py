from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict


class VerifyRequest(BaseModel):
    user_id: Optional[str] = None
    image: str
    require_liveness: bool = False
    # Active challenge liveness (online access control):
    #   liveness_mode="active" + liveness_token — proof из /liveness/challenge/stream.
    #   При active token валидируется+consumes (single-use), passive не запускается,
    #   ответ получает liveness_passed=True (active-proven).
    liveness_mode: str = "passive"  # "passive" | "active"
    liveness_token: Optional[str] = None


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

    # True при match с низким margin («серая» зона) — клиенту рекомендуем
    # active-challenge liveness (turn/nod) через WS-стрим для подтверждения.
    challenge_recommended: Optional[bool] = None

    # Liveness-диагностика: сырой real_score (softmax[idx1] yakhyo MiniFASNet).
    liveness_score: Optional[float] = None

    # Честные бинарные per-class вероятности: real_prob=softmax[idx1], spoof_prob=
    # softmax[idx2]. idx0 (мёртвый класс) не выносим. Модель эффективно бинарная и
    # НЕ различает print/replay/cutout — поэтому per-attack-type меток здесь нет
    # (см. memory liveness-yakhyo-logit-semantics). Заполняется при require_liveness.
    spoofing_indicators: Optional[dict[str, Any]] = None

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
