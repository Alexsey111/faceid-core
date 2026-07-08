# app/api/_helpers.py — общие хелперы для роутов (DRY: раньше дублировались).
#
# Содержит то, что копировалось между роут-модулями с риском рассинхрона:
#   - MAX_IMAGE_SIZE       (дублировался в upload.py и verify.py)
#   - decode_image_bytes   (дублировался в liveness.py и liveness_challenge.py
#                           с поведенческой вариацией raise vs None)
#   - get_request_id       (байт-идентично в verify.py и verify_async.py)

import time
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi import Request

# Лимит размера загружаемого изображения (5 MiB). Раньше дублировался в
# upload.py и verify.py → риск рассинхрона лимита между /upload и /verify.
MAX_IMAGE_SIZE = 5 * 1024 * 1024


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Декодирует байты в BGR uint8 кадр (без resize — checker сам делает кроп).
    Raise ValueError если байты не декодируются. Раньше дублировалось в
    liveness.py (_decode, raise) и liveness_challenge.py (_decode_jpeg, None) —
    вариация контракта была миной для copy-paste в новые роуты."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image")
    return image


def get_request_id(request: Request) -> str:
    """Возвращает request_id из request.state, генерирует req-<ms> если нет.
    Раньше байт-идентично дублировалось в verify.py и verify_async.py —
    расхождение формата ломало трассировку инцидентов по логам."""
    state = getattr(request, "state", None)
    if state is None:
        state = SimpleNamespace()
        try:
            setattr(request, "state", state)
        except Exception:
            pass

    request_id = getattr(state, "request_id", None)
    if request_id:
        return request_id

    request_id = f"req-{int(time.time() * 1000)}"
    try:
        state.request_id = request_id
    except Exception:
        pass
    return request_id