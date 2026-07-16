from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.core.config import settings
from app.ml.detection.retinaface_detector import RetinaFaceDetector
from app.ml.liveness.scoring import score_image_liveness
from app.ml.runtime import get_liveness_checker
from app.api._helpers import MAX_IMAGE_SIZE, decode_image_bytes
from app.services.rate_limiter import RateLimiter

router = APIRouter()


@router.post("/liveness")
async def check_liveness(
    request: Request,
    file: UploadFile = File(...),
):
    """Standalone passive liveness-проверка (ТЗ 4: POST /api/v1/liveness).

    Контракт ответа (ТЗ): {is_live, confidence, spoofing_indicators}.
    face_detected — доп-диагностика (полезно клиенту для retry-логики).
    Для high-security допуска используется active-challenge
    (/api/v1/liveness/challenge/*), не этот passive-роут.
    """
    # Rate-limit parity с /verify (ТЗ 3.1: защита эндпоинтов).
    RateLimiter.check(request, "liveness", limit=10)

    if not settings.LIVENESS_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Liveness is disabled",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image format")

    image_bytes = await file.read()

    # Size-validation parity с /verify (MAX_IMAGE_SIZE=5 MiB) — защита от
    # oversized-payload DoS.
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="Image too large")

    try:
        image = decode_image_bytes(image_bytes)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image")

    checker = get_liveness_checker()
    if checker is None:
        raise HTTPException(
            status_code=503,
            detail="Liveness model is not available",
        )

    # RetinaFaceDetector дёшев: тяжёлая модель детекции кэширована в runtime
    # (get_face_app_for_size, @lru_cache). det_size=SMALL (320) — для latency.
    detector = RetinaFaceDetector(det_size=settings.RETINA_DET_SIZE_SMALL)
    result = score_image_liveness(
        image, detector, checker, settings.LIVENESS_THRESHOLD
    )

    # ТЗ-контракт: is_live (булево решение) + confidence (real_score ∈ [0,1]).
    # Старые ключи liveness/score убраны — никто не парсил (app/demo используют
    # active-challenge). face_detected оставлен как доп-диагностика.
    return {
        "is_live": result["liveness"],
        "confidence": result["score"],
        "face_detected": result["face_detected"],
        "spoofing_indicators": result.get(
            "spoofing_indicators", {"real_prob": 0.0, "spoof_prob": 0.0}
        ),
    }