from fastapi import APIRouter, File, HTTPException, UploadFile
import cv2
import numpy as np

from app.core.config import settings
from app.ml.detection.retinaface_detector import RetinaFaceDetector
from app.ml.liveness.scoring import score_image_liveness
from app.ml.runtime import get_liveness_checker

router = APIRouter()


def _decode(image_bytes: bytes) -> np.ndarray:
    """Декодирует байты в BGR uint8 кадр (без resize — checker сам делает кроп)."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image")
    return image


@router.post("/liveness")
async def check_liveness(file: UploadFile = File(...)):
    try:
        if not settings.LIVENESS_ENABLED:
            raise HTTPException(
                status_code=503,
                detail="Liveness is disabled",
            )

        image_bytes = await file.read()
        image = _decode(image_bytes)

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

        return {
            "liveness": result["liveness"],
            "score": result["score"],
            "face_detected": result["face_detected"],
            "spoofing_indicators": result.get(
                "spoofing_indicators", {"real_prob": 0.0, "spoof_prob": 0.0}
            ),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))