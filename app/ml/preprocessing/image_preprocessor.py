# app/ml/preprocessing/image_preprocessor.py - Предобработка изображений

import cv2
import numpy as np

from app.core.config import settings


class ImagePreprocessor:
    """
    Prepares input image for ML pipeline.
    """

    MAX_SIZE = 480

    def __init__(self, max_side: int | None = None):
        self.max_side = max_side or settings.PREPROCESS_MAX_SIDE or self.MAX_SIZE

    def decode(self, image_bytes: bytes) -> np.ndarray:
        """
        Decode bytes into OpenCV image.
        """
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

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
