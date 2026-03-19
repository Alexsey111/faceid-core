# app/ml/detection/retinaface_detector.py

from typing import Dict, Any, List, Optional
import numpy as np

from app.ml.runtime import get_face_app


class RetinaFaceDetector:
    """
    Face detection module using InsightFace.
    """

    def __init__(self):

        self.app = get_face_app()

    def detect(self, image: np.ndarray) -> Optional[List[Dict[str, Any]]]:
        """
        Detect faces in image.

        Returns:
            List of {
                bbox: [x1, y1, x2, y2],
                landmarks: [[x,y]...],
                embedding: np.ndarray
            }
        """

        faces = self.app.get(image)

        if not faces:
            return None

        # Convert Face objects to dicts
        return [
            {
                "bbox": face.bbox.astype(float).tolist(),
                "landmarks": face.kps.tolist(),
                "embedding": face.embedding
            }
            for face in faces
        ]