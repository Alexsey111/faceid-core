from fastapi import APIRouter, File, HTTPException, UploadFile
import cv2
import numpy as np

from app.ml.runtime import get_liveness_model

router = APIRouter()

THRESHOLD = 0.7


def preprocess(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image")

    image = cv2.resize(image, (128, 128))
    image = image.astype("float32") / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return image


@router.post("/liveness")
async def check_liveness(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = preprocess(image_bytes)

        session = get_liveness_model()
        if session is None:
            raise HTTPException(
                status_code=503,
                detail="Liveness model is not available",
            )

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: image})
        score = float(outputs[0][0][1])

        return {
            "liveness": score > THRESHOLD,
            "score": score,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
