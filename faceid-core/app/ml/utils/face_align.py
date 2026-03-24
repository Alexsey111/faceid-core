import cv2
import numpy as np


_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(image: np.ndarray, landmarks, output_size: int = 112) -> np.ndarray:
    src = np.array(landmarks, dtype=np.float32)

    if src.shape != (5, 2):
        raise ValueError("Invalid landmarks shape")

    transform = cv2.estimateAffinePartial2D(src, _ARCFACE_DST)[0]
    if transform is None:
        raise ValueError("Failed to estimate face alignment transform")

    aligned = cv2.warpAffine(image, transform, (output_size, output_size))
    return aligned