# app/ml/preprocessing/image_preprocessor.py - Предобработка изображений

import cv2
import numpy as np


class ImagePreprocessor:
    """
    Prepares input image for ML pipeline.
    """

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

        Rules:
        - max side > 2000 px -> resize to 960
        - max side > 1200 px -> resize to 640
        - otherwise keep original
        """
        height, width = image.shape[:2]
        max_side = max(height, width)

        if max_side > 2000:
            target_max_side = 960
        elif max_side > 1200:
            target_max_side = 640
        else:
            return image

        scale = target_max_side / float(max_side)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    def to_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Convert BGR -> RGB.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to float32.
        """
        image = image.astype("float32")
        return image / 255.0

    def process(self, image_bytes: bytes) -> np.ndarray:
        """
        Full preprocessing pipeline.
        Returns BGR image (OpenCV format) in 0-255 range.
        """
        image = self.decode(image_bytes)
        image = self.resize_if_needed(image)

        # InsightFace expects BGR image in 0-255 range
        # Don't normalize or convert to RGB here
        return image