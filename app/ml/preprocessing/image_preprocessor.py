# app/ml/preprocessing/image_preprocessor.py - Предобработка изображений

import io

import cv2
import numpy as np

from app.core.config import settings

# HEIC/HEIF/AVIF не поддерживаются OpenCV — fallback через PIL+pillow-heif.
# Регистрация opener'а ленивая (один раз), отсутствие пакета — graceful退化 в ValueError.
_HEIF_REGISTERED: bool = False


def _ensure_heif_support() -> bool:
    """Зарегистрировать pillow-heif как PIL opener (lazy, single-use)."""
    global _HEIF_REGISTERED
    if _HEIF_REGISTERED:
        return True
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        _HEIF_REGISTERED = True
        return True
    except Exception:
        return False


def _decode_via_pil(image_bytes: bytes) -> np.ndarray | None:
    """Fallback-декодер через PIL+pillow-heif (HEIC/HEIF/AVIF и др.).

    Возвращает BGR uint8 (контракт OpenCV-пайплайна) или None при ошибке/
    отсутствии pillow-heif.
    """
    if not _ensure_heif_support():
        return None
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)              # (H, W, 3) RGB
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


class ImagePreprocessor:
    """
    Prepares input image for ML pipeline.
    """

    MAX_SIZE = 480

    def __init__(self, max_side: int | None = None):
        self.max_side = max_side or settings.PREPROCESS_MAX_SIDE or self.MAX_SIZE

    def decode(self, image_bytes: bytes) -> np.ndarray:
        """
        Decode bytes into OpenCV image (BGR uint8).

        Сначала cv2.imdecode (быстрый путь для JPEG/PNG/WebP/BMP/TIFF).
        Если OpenCV не смог (HEIC/HEIF/AVIF и прочие неподдерживаемые) —
        fallback через PIL+pillow-heif с конвертацией RGB→BGR.
        """
        np_arr = np.frombuffer(image_bytes, np.uint8) if image_bytes else np.empty(0, np.uint8)
        try:
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except cv2.error:
            # пустой/битый буфер уходит в fallback (потенциально PIL-путь)
            image = None

        if image is None:
            # cv2 не поддерживает HEIC/HEIF/AVIF — fallback через PIL+pillow-heif
            image = _decode_via_pil(image_bytes)

        if image is None:
            raise ValueError("Invalid image file")

        return image

    def resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        """
        Resize large images to reduce face detection cost.

        Keep the long side under `MAX_SIZE` so detector work stays bounded.
        """
        h, w = image.shape[:2]
        max_side = max(h, w)

        scale = self.max_side / float(max_side)
        if scale >= 1.0:
            return image

        new_width = max(1, int(w * scale))
        new_height = max(1, int(h * scale))

        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    def process(self, image_bytes: bytes) -> np.ndarray:
        """
        Full preprocessing pipeline.
        Returns BGR image (OpenCV format) in 0-255 range.
        """
        image = self.decode(image_bytes)
        return self.process_image(image)

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline for already decoded images.
        Returns BGR image (OpenCV format) in 0-255 range.
        """
        image = self.resize_if_needed(image)

        # InsightFace expects BGR image in 0-255 range
        # Don't normalize or convert to RGB here
        return image
