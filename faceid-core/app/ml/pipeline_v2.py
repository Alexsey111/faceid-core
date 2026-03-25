from pathlib import Path
from typing import Dict, Any, Optional
import os
import time

import cv2
import numpy as np

from app.core.config import settings
from app.ml.dependencies import get_batch_encoder
from app.ml.preprocessing.image_preprocessor import ImagePreprocessor
from app.ml.detection.fast_detector import FastFaceDetector
from app.ml.detection.retinaface_detector import RetinaFaceDetector
from app.ml.utils.face_align import align_face


class FacePipelineV2:
    """
    Production-oriented V2 pipeline:

    1. preprocess
    2. fast detector
    3. fallback to RetinaFace if fast detector finds nothing
    4. crop/align
    5. ONNX ArcFace encode
    """

    FAST_CONFIDENCE_THRESHOLD = 0.6
    FAST_BBOX_EXPAND_SCALE = 0.30

    def __init__(self):
        print(f"PID={os.getpid()} pipeline_v2 created")
        self._initialized = False

        self.preprocessor = ImagePreprocessor()

        self.fast_detector: Optional[FastFaceDetector] = None
        self.detector: Optional[RetinaFaceDetector] = None
        self.encoder: Optional[object] = None

    def _init(self):
        if not self._initialized:
            fast_model, fast_config = self._resolve_fast_detector_paths()

            self.fast_detector = FastFaceDetector(
                model_path=str(fast_model),
                config_path=str(fast_config),
            )
            self.detector = RetinaFaceDetector()
            self.encoder = get_batch_encoder()
            self._initialized = True

    def process(self, image_bytes: bytes) -> Dict[str, Any]:
        self._init()

        assert self.fast_detector is not None, "fast_detector not initialized"
        assert self.detector is not None, "retinaface detector not initialized"
        assert self.encoder is not None, "encoder not initialized"

        timings: Dict[str, float] = {}

        t0 = time.time()
        image = self.preprocessor.process(image_bytes)
        timings["preprocess_ms"] = (time.time() - t0) * 1000

        t0 = time.time()
        fast_faces = self.fast_detector.detect(image)
        timings["fast_detect_ms"] = (time.time() - t0) * 1000

        detection: Dict[str, Any] | None = None
        face_input: np.ndarray | None = None
        bbox_source = "fast"

        if len(fast_faces) == 1:
            x1, y1, x2, y2, confidence = fast_faces[0]

            if confidence < self.FAST_CONFIDENCE_THRESHOLD:
                raise ValueError("Face not detected")

            h, w = image.shape[:2]
            x1, y1, x2, y2 = self._expand_bbox(
                x1, y1, x2, y2, w, h, self.FAST_BBOX_EXPAND_SCALE
            )

            roi = self._safe_crop(image, x1, y1, x2, y2)
            detections = self.detector.detect(roi)

            if detections and len(detections) == 1:
                det = detections[0]

                if det.get("landmarks") is not None:
                    face_input = align_face(roi, det["landmarks"])
                    bbox_source = "fast+retina"
                else:
                    face_input = roi
                    bbox_source = "fast_crop"
            else:
                face_input = roi
                bbox_source = "fast_only"

            detection = {
                "bbox": [x1, y1, x2, y2],
                "landmarks": None,
                "confidence": float(confidence),
            }

        elif len(fast_faces) == 0:
            t0 = time.time()
            detections = self.detector.detect(image)
            timings["fallback_detect_ms"] = (time.time() - t0) * 1000

            if not detections:
                raise ValueError("Face not detected")

            if len(detections) > 1:
                raise ValueError("Multiple faces not allowed")

            detection = detections[0]
            bbox_source = "retinaface"

            x1, y1, x2, y2 = map(int, detection["bbox"])

            if detection.get("landmarks") is not None:
                face_input = align_face(image, detection["landmarks"])
            else:
                face_input = self._safe_crop(image, x1, y1, x2, y2)

        else:
            raise ValueError("Multiple faces not allowed")

        if detection is None or face_input is None:
            raise ValueError("Failed to prepare face input")

        if face_input.size == 0:
            raise ValueError("Empty face crop")

        if face_input.shape[:2] != (112, 112):
            face_input = cv2.resize(face_input, (112, 112))

        t0 = time.time()
        embedding = self.encoder.encode(face_input)
        norm = np.linalg.norm(embedding)
        if not (0.99 <= norm <= 1.01):
            embedding = embedding / (norm + 1e-8)
        timings["encode_ms"] = (time.time() - t0) * 1000

        timings["total_pipeline_ms"] = sum(timings.values())

        return {
            "embedding": embedding,
            "bbox": detection["bbox"],
            "landmarks": detection.get("landmarks"),
            "bbox_source": bbox_source,
            "timings": timings,
        }

    def _resolve_fast_detector_paths(self):
        project_root = Path(__file__).resolve().parents[3]
        candidates = [
            Path(settings.MODELS_DIR) / "fast_detector",
            Path(settings.MODELS_DIR) / "models" / "fast_detector",
            project_root / "models" / "fast_detector",
        ]

        for candidate in candidates:
            model_path = candidate / "res10_300x300_ssd_iter_140000.caffemodel"
            config_path = candidate / "deploy.prototxt"
            if model_path.exists() and config_path.exists():
                return model_path, config_path

        raise FileNotFoundError("Fast detector files not found")

    def _expand_bbox(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        image_width: int,
        image_height: int,
        scale: float,
    ) -> tuple[int, int, int, int]:
        bw = x2 - x1
        bh = y2 - y1

        x1 = max(0, int(x1 - bw * scale))
        y1 = max(0, int(y1 - bh * scale))
        x2 = min(image_width, int(x2 + bw * scale))
        y2 = min(image_height, int(y2 + bh * scale))

        return x1, y1, x2, y2

    def _safe_crop(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> np.ndarray:
        crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            raise ValueError("Empty face crop")
        return crop
