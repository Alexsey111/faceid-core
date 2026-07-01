# app/ml/detection/retinaface_detector.py

import logging
from typing import Dict, Any, List, Optional

import numpy as np

from app.ml.runtime import get_face_app_for_size

logger = logging.getLogger(__name__)


class RetinaFaceDetector:
    """
    Face detection module using InsightFace (SCRFD, buffalo_l).

    Возвращает 5-point landmarks, необходимые для аффинного выравнивания
    лица (insightface.utils.face_align.norm_crop) перед ArcFace-encode.
    """

    def __init__(self, det_size: int):
        self.app = get_face_app_for_size(int(det_size))
        self.det_model = self.app.det_model

    def detect(self, image: np.ndarray) -> Optional[List[Dict[str, Any]]]:
        """
        Detect faces in image using a single SCRFD detector.

        Returns:
            Список детекций, отсортированный по confidence убыванию
            (так чтобы caller мог брать top-1 как faces[0]). Элемент:
            {
                bbox: [x1, y1, x2, y2],      # xyxy, float, абсолютные пиксели
                landmarks: [[x, y], ...] | None,  # 5-point kpss, абсолютные пиксели
                confidence: float,
            }
            None — если лиц нет.
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
                    "confidence": float(bboxes[idx, 4]),
                }
            )

        # top-1 по score — чтобы V2 брал наиболее уверенное лицо
        faces.sort(key=lambda d: d["confidence"], reverse=True)
        return faces

    def detect_batch(
        self, images: list[np.ndarray]
    ) -> list[list[Dict[str, Any]]]:
        """
        Побатчевая детекция. У SCRFD (insightface) нет batch-API, поэтому
        детектим по одному изображению. Длина результата всегда равна
        len(images): для изображения без лиц (или упавшей детекции) — пустой
        список, чтобы caller'ы, проверяющие len(faces)==0, корректно получали
        "Face not detected".
        """
        results: list[list[Dict[str, Any]]] = []
        for image in images:
            try:
                faces = self.detect(image)
            except Exception as exc:
                logger.warning("detect_batch item failed: %s", exc)
                results.append([])
                continue
            results.append(faces if faces is not None else [])
        return results
