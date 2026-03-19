# face_alignment.py - Выравнивание лица

# app/ml/alignment/face_alignment.py

import cv2
import numpy as np


class FaceAlignment:
    """
    Aligns face using 5 facial landmarks.
    """

    def __init__(self):

        # стандартные точки ArcFace
        self.reference = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )

        self.output_size = (112, 112)

    def align(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Align face using similarity transform.
        """

        src = landmarks.astype(np.float32)
        dst = self.reference

        transform, _ = cv2.estimateAffinePartial2D(src, dst)

        aligned = cv2.warpAffine(
            image,
            transform,
            self.output_size,
            borderValue=0.0
        )

        return aligned