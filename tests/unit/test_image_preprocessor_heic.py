# test_image_preprocessor_heic.py — HEIC/HEIF fallback в ImagePreprocessor.decode.
#
# cv2.imdecode не поддерживает HEIC → decode должен упасть в PIL+pillow-heif fallback
# и вернуть BGR uint8. Генерируем HEIC из массива через pillow_heif, кодируем в bytes,
# декодируем через ImagePreprocessor.decode и сравниваем с исходным изображением
# (с точностью до потерь HEIC). Также: JPEG/PNG идут через cv2 fast-path; мусор → ValueError.
from __future__ import annotations

import io

import cv2
import numpy as np
import pytest

from app.ml.preprocessing.image_preprocessor import ImagePreprocessor

_unit = pytest.mark.unit

try:
    import pillow_heif  # noqa: F401
    _HEIF_OK = True
except Exception:
    _HEIF_OK = False

_skip_no_heif = pytest.mark.skipif(not _HEIF_OK, reason="pillow-heif не установлен")


def _make_test_image() -> np.ndarray:
    """RGB-градиент 32×32 (ненулевой, цветной) → конвертируем в BGR для сравнения."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, 32, dtype=np.uint8)[None, :]  # R
    img[:, :, 1] = np.linspace(0, 255, 32, dtype=np.uint8)[:, None]  # G
    img[:, :, 2] = 100  # B
    return img  # RGB


def _encode_heic(rgb: np.ndarray) -> bytes:
    from PIL import Image
    pillow_heif.register_heif_opener()
    pil = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="HEIF", quality=90)
    return buf.getvalue()


@_unit
def test_decode_jpeg_via_cv2_fastpath():
    """JPEG декодируется cv2 (fast-path), fallback не нужен."""
    img = np.full((24, 24, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    decoded = ImagePreprocessor().decode(buf.tobytes())
    assert decoded.shape == (24, 24, 3)
    assert decoded.dtype == np.uint8


@_unit
@_skip_no_heif
def test_decode_heic_via_pil_fallback():
    """HEIC не поддерживается cv2 → fallback через PIL+pillow-heif → BGR uint8."""
    rgb = _make_test_image()
    heic_bytes = _encode_heic(rgb)
    # sanity: cv2 один не справляется
    assert cv2.imdecode(np.frombuffer(heic_bytes, np.uint8), cv2.IMREAD_COLOR) is None

    decoded = ImagePreprocessor().decode(heic_bytes)
    assert decoded.shape == (32, 32, 3)
    assert decoded.dtype == np.uint8
    # BGR-контракт: decode должен вернуть BGR. Сравним с ожидаемым BGR с допуском на HEIC-потери.
    expected_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    assert np.mean(np.abs(decoded.astype(int) - expected_bgr.astype(int))) < 15.0


@_unit
@_skip_no_heif
def test_decode_heic_preserves_color_channels_order():
    """Защита от инверсии RGB/BGR: красный пиксель → BGR должен быть (B=low, G=low, R=high)."""
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255  # чистый красный в RGB
    heic_bytes = _encode_heic(rgb)
    decoded = ImagePreprocessor().decode(heic_bytes)
    # в BGR красный → канал 2 высокий, каналы 0/1 низкие
    assert decoded[8, 8, 2] > 180, f"красный должен попасть в BGR-канал 2, got {decoded[8, 8]}"
    assert decoded[8, 8, 0] < 75 and decoded[8, 8, 1] < 75


@_unit
def test_decode_garbage_raises_value_error():
    with pytest.raises(ValueError):
        ImagePreprocessor().decode(b"not an image at all")


@_unit
def test_decode_empty_raises_value_error():
    with pytest.raises(ValueError):
        ImagePreprocessor().decode(b"")