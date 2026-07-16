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


def extract_client_ip(request: Request) -> str | None:
    """Извлечь реальный клиентский IP из заголовков reverse-proxy/LB.

    За nginx (api_lb) request.client.host — это IP прокси. Сначала X-Real-IP
    (nginx ставит), затем первая запись X-Forwarded-For, затем fallback
    request.client.host. Тот же паттерн, что в rate_limiter.py:40-44 (DRY).

    Робастна к mock-объектам в тестах (SimpleNamespace без .headers/.client) —
    getattr-проверки как в get_request_id. Возвращает None, если заголовков нет.
    """
    headers = getattr(request, "headers", None)
    client_ip: str | None = None
    if headers is not None:
        client_ip = headers.get("X-Real-IP")
        if not client_ip:
            forwarded = headers.get("X-Forwarded-For")
            if forwarded:
                client_ip = forwarded.split(",")[0].strip()
    if not client_ip:
        # fallback на transport-IP (для dev без proxy = реальный клиент;
        # за proxy = IP прокси, что хуже, но лучше чем None).
        client = getattr(request, "client", None)
        if client is not None:
            host = getattr(client, "host", None)
            if host:
                client_ip = host
    return client_ip or None