# app/ml/detection/retinaface_detector.py

from typing import Dict, Any, List, Optional
import numpy as np

from app.ml.runtime import get_face_app_for_size


class RetinaFaceDetector:
    """
    Face detection module using InsightFace.
    """

    def __init__(self, det_size: int):
        self.app = get_face_app_for_size(int(det_size))
        self.det_model = self.app.det_model

    def detect(self, image: np.ndarray) -> Optional[List[Dict[str, Any]]]:
        """
        Detect faces in image using a single SCRFD detector.

        Returns:
            List of {
                bbox: [x1, y1, x2, y2],
                landmarks: [[x,y]...],
            }
        """
        bboxes, kpss = self.det_model.detect(
            image,
            max_num=0,
            metric="default",
        )

        if bboxes is None or bboxes.shape[0] == 0:
            return None

        faces: list[dict[str, Any]] = []
        for idx in range(bboxes.shape[0]):
            bbox = bboxes[idx, 0:4].astype(float).tolist()
            landmarks = None
            if kpss is not None:
                landmarks = kpss[idx].astype(float).tolist()

            faces.append(
                {
                    "bbox": bbox,
                    "landmarks": landmarks,
                }
            )

        return faces
