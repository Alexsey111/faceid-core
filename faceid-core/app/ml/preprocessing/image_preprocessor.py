# image_preprocessor.py - Предобработка изображений

# app/ml/preprocessing/image_preprocessor.py

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

        # InsightFace expects BGR image in 0-255 range
        # Don't normalize or convert to RGB here

        return image