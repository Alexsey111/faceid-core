# app/ml/pipeline.py

from pathlib import Path
from typing import Dict, Any, Optional
import os
import cv2
import numpy as np
import time

from app.core.config import settings
from app.ml.preprocessing.image_preprocessor import ImagePreprocessor
from app.ml.detection.retinaface_detector import RetinaFaceDetector
from app.ml.liveness.antispoof_model import AntiSpoofModel
from app.ml.liveness.onnx_liveness import OnnxLivenessChecker
from app.ml.liveness.model_paths import resolve_liveness_model_path
from app.ml.embedding.arcface_encoder import ArcFaceEncoder


class FacePipeline:

    def __init__(self):
        print(f"PID={os.getpid()} pipeline created")
        self._initialized = False

        # Lightweight, init immediately
        self.preprocessor = ImagePreprocessor()

        # Lazy init
        self.detector: Optional[RetinaFaceDetector] = None
        self.encoder: Optional[ArcFaceEncoder] = None
        self.liveness: Optional[AntiSpoofModel] = None
        self.liveness_checker: Optional[OnnxLivenessChecker] = None

    def _init(self):
        if not self._initialized:
            self.detector = RetinaFaceDetector()
            self.encoder = ArcFaceEncoder()

            # Liveness model is optional
            model_path = resolve_liveness_model_path(settings.MODELS_DIR)
            if model_path is not None:
                self.liveness = AntiSpoofModel(str(model_path))

            if model_path is not None:
                self.liveness_checker = OnnxLivenessChecker(
                    model_path=str(model_path),
                    threshold=settings.LIVENESS_THRESHOLD,
                )

            self._initialized = True

    def process(self, image_bytes: bytes) -> Dict[str, Any]:

        self._init()

        assert self.detector is not None, "detector not initialized"

        timings: Dict[str, float] = {}

        t0 = time.time()
        image = self.preprocessor.process(image_bytes)
        timings["preprocess_ms"] = (time.time() - t0) * 1000

        t0 = time.time()
        detections = self.detector.detect(image)
        timings["detect_ms"] = (time.time() - t0) * 1000

        if not detections:
            raise ValueError("Face not detected")

        if len(detections) > 1:
            raise ValueError("Multiple faces not allowed")

        detection = detections[0]

        assert self.encoder is not None, "encoder not initialized"

        t0 = time.time()
        embedding = self.encoder.normalize(detection["embedding"])
        timings["encode_ms"] = (time.time() - t0) * 1000

        # --- LIVENESS ---
        t0 = time.time()

        x1, y1, x2, y2 = map(int, detection["bbox"])
        face_crop = image[y1:y2, x1:x2]

        passive_score: Optional[float] = None
        if self.liveness and settings.DEBUG:
            passive_score = float(self.liveness.predict(face_crop))

        texture_score = self._texture_analysis(face_crop)
        blur_score = self._blur_score(face_crop)

        timings["liveness_ms"] = (time.time() - t0) * 1000
        timings["total_pipeline_ms"] = sum(timings.values())

        return {
            "embedding": embedding,
            "bbox": detection["bbox"],
            "landmarks": detection["landmarks"],
            "liveness": {
                "passive": passive_score,
                "texture": texture_score,
                "blur": blur_score
            },
            "timings": timings
        }

    def _texture_analysis(self, face_crop: np.ndarray) -> float:
        """
        Analyze face texture for screen/print detection.
        High frequency components indicate moire/screen artifacts.
        Returns score 0-1 where higher = more texture (likely real).
        """
        if face_crop.size == 0:
            return 0.0

        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            return 0.0

        # High frequency components (moire / screen artifacts)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        score = laplacian.var()

        # Normalize
        return float(min(score / 1000.0, 1.0))

    def _blur_score(self, face_crop: np.ndarray) -> float:
        """
        Detect blur using variance of Laplacian.
        Higher score = sharper image (less blur).
        Returns score 0-1 where higher = sharper.
        """
        if face_crop.size == 0:
            return 0.0

        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            return 0.0

        var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Normalize
        return float(min(var / 500.0, 1.0))
