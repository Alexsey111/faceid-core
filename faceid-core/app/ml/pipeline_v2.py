# faceid-core\app\ml\pipeline_v2.py

from pathlib import Path
from typing import Dict, Any, Optional
import logging
import os
import time

import cv2
import numpy as np

from app.core.config import settings
from app.ml.batch_encoder import BatchEncoder
from app.ml.dependencies import get_batch_encoder
from app.ml.embedding.onnx_arcface_encoder import OnnxArcFaceEncoder
from app.ml.liveness.onnx_liveness import OnnxLivenessChecker
from app.ml.liveness.model_paths import resolve_liveness_model_path
from app.ml.preprocessing.image_preprocessor import ImagePreprocessor
from app.ml.quality.image_quality_gate import ImageQualityGate
from app.ml.detection.fast_detector import FastFaceDetector
from app.ml.detection.retinaface_detector import RetinaFaceDetector
from app.ml.utils.face_align import align_face


logger = logging.getLogger(__name__)


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
        self.quality_gate = ImageQualityGate()

        self.fast_detector: Optional[FastFaceDetector] = None
        self.detector: Optional[RetinaFaceDetector] = None
        self.encoder: BatchEncoder | OnnxArcFaceEncoder | None = None
        self.liveness_checker: Optional[OnnxLivenessChecker] = None

    def _init(self):
        if not self._initialized:
            fast_model, fast_config = self._resolve_fast_detector_paths()

            self.fast_detector = FastFaceDetector(
                model_path=str(fast_model),
                config_path=str(fast_config),
            )
            self.detector = RetinaFaceDetector()
            self.encoder = get_batch_encoder()
            if settings.LIVENESS_ENABLED:
                liveness_path = resolve_liveness_model_path(settings.MODELS_DIR)

                if liveness_path is not None:
                    logger.info("liveness model loaded from %s", liveness_path)
                    self.liveness_checker = OnnxLivenessChecker(
                        str(liveness_path),
                        threshold=settings.LIVENESS_THRESHOLD,
                    )
                else:
                    logger.warning("liveness model not found, disabled")
                    self.liveness_checker = None
            self._initialized = True

    def _prepare_face_from_detection(
        self,
        image: np.ndarray,
        image_quality: Any,
        fast_faces: list[list[float]],
        timings: Dict[str, float],
    ) -> Dict[str, Any]:
        detector = self.detector
        assert detector is not None, "retinaface detector not initialized"

        detection: Dict[str, Any] | None = None
        face_input: np.ndarray | None = None
        bbox_source = "fast"

        if len(fast_faces) == 1:
            x1_f, y1_f, x2_f, y2_f, confidence = fast_faces[0]
            x1, y1, x2, y2 = map(int, (x1_f, y1_f, x2_f, y2_f))

            h, w = image.shape[:2]
            x1, y1, x2, y2 = self._expand_bbox(
                x1, y1, x2, y2, w, h, self.FAST_BBOX_EXPAND_SCALE
            )

            roi = self._safe_crop(image, x1, y1, x2, y2)

            if roi.mean() < 5:
                raise ValueError("bad crop")

            if confidence >= self.FAST_CONFIDENCE_THRESHOLD:
                face_input = cv2.resize(roi, (112, 112))
                bbox_source = "fast_only"
            else:
                raise ValueError("Face not detected")

            detection = {
                "bbox": [x1, y1, x2, y2],
                "landmarks": None,
                "confidence": float(confidence),
            }

        elif len(fast_faces) == 0:
            t0 = time.time()
            detections = detector.detect(image)
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

        # ----------------------------
        # POST-DETECT QUALITY GATE
        # ----------------------------
        t0 = time.time()
        face_quality = self.quality_gate.evaluate_detection(
            bbox=detection["bbox"],
            landmarks=detection.get("landmarks"),
        )
        timings["quality_gate_face_ms"] = (time.time() - t0) * 1000

        if not face_quality.passed:
            timings["total_pipeline_ms"] = sum(timings.values())
            return {
                "status": "quality_reject",
                "quality_reason": face_quality.reason,
                "quality_details": face_quality.details,
                "bbox": detection["bbox"],
                "landmarks": detection.get("landmarks"),
                "bbox_source": bbox_source,
                "timings": timings,
            }

        if face_input.size == 0:
            raise ValueError("Empty face crop")

        if face_input.shape[:2] != (112, 112):
            face_input = cv2.resize(face_input, (112, 112))

        if face_input.mean() < 5:
            raise ValueError("Invalid face crop (too dark)")

        face_input = np.ascontiguousarray(face_input, dtype=np.uint8)

        liveness_passed = None
        liveness_score = None
        if settings.LIVENESS_ENABLED and self.liveness_checker:
            try:
                t0 = time.time()
                liveness_passed, liveness_score = self.liveness_checker.predict(face_input)
                timings["liveness_ms"] = (time.time() - t0) * 1000

                if liveness_passed is False:
                    timings["total_pipeline_ms"] = sum(timings.values())

                    return {
                        "status": "spoof",
                        "liveness_passed": False,
                        "liveness_score": liveness_score,
                        "bbox": detection["bbox"],
                        "landmarks": detection.get("landmarks"),
                        "bbox_source": bbox_source,
                        "timings": timings,
                    }

            except Exception as e:
                logger.warning("liveness_error: %s", str(e))
                liveness_passed = None
                liveness_score = None

        return {
            "status": "ok",
            "face_input": face_input,
            "bbox": detection["bbox"],
            "landmarks": detection.get("landmarks"),
            "liveness_passed": liveness_passed,
            "liveness_score": liveness_score,
            "bbox_source": bbox_source,
            "quality_details": {
                **image_quality.details,
                **face_quality.details,
            },
            "timings": timings,
        }

    def prepare_face_input(self, image_bytes: bytes) -> Dict[str, Any]:
        self._init()

        assert self.fast_detector is not None, "fast_detector not initialized"
        assert self.detector is not None, "retinaface detector not initialized"
        assert self.encoder is not None, "encoder not initialized"

        timings: Dict[str, float] = {}

        t0 = time.time()
        image = self.preprocessor.process(image_bytes)
        timings["preprocess_ms"] = (time.time() - t0) * 1000

        t0 = time.time()
        image_quality = self.quality_gate.evaluate_image(image)
        timings["quality_gate_pre_ms"] = (time.time() - t0) * 1000

        if not image_quality.passed:
            timings["total_pipeline_ms"] = sum(timings.values())
            return {
                "status": "quality_reject",
                "quality_reason": image_quality.reason,
                "quality_details": image_quality.details,
                "timings": timings,
            }

        t0 = time.time()
        fast_faces = self.fast_detector.detect(image)
        timings["fast_detect_ms"] = (time.time() - t0) * 1000

        return self._prepare_face_from_detection(image, image_quality, fast_faces, timings)

    def prepare_face_input_from_image(self, image: np.ndarray) -> Dict[str, Any]:
        self._init()

        assert self.fast_detector is not None, "fast_detector not initialized"
        assert self.detector is not None, "retinaface detector not initialized"
        assert self.encoder is not None, "encoder not initialized"

        timings: Dict[str, float] = {}

        t0 = time.time()
        image = self.preprocessor.process_image(image)
        timings["preprocess_ms"] = (time.time() - t0) * 1000

        t0 = time.time()
        image_quality = self.quality_gate.evaluate_image(image)
        timings["quality_gate_pre_ms"] = (time.time() - t0) * 1000

        if not image_quality.passed:
            timings["total_pipeline_ms"] = sum(timings.values())
            return {
                "status": "quality_reject",
                "quality_reason": image_quality.reason,
                "quality_details": image_quality.details,
                "timings": timings,
            }

        t0 = time.time()
        fast_faces = self.fast_detector.detect(image)
        timings["fast_detect_ms"] = (time.time() - t0) * 1000

        return self._prepare_face_from_detection(image, image_quality, fast_faces, timings)

    def prepare_face_inputs(self, image_bytes_list: list[bytes]) -> list[Dict[str, Any]]:
        self._init()

        assert self.fast_detector is not None, "fast_detector not initialized"

        images: list[np.ndarray] = []
        image_qualities: list[Any] = []
        timings_list: list[Dict[str, float]] = []
        results: list[Dict[str, Any]] = []

        for image_bytes in image_bytes_list:
            timings: Dict[str, float] = {}
            t0 = time.time()
            image = self.preprocessor.process(image_bytes)
            timings["preprocess_ms"] = (time.time() - t0) * 1000

            t0 = time.time()
            image_quality = self.quality_gate.evaluate_image(image)
            timings["quality_gate_pre_ms"] = (time.time() - t0) * 1000

            images.append(image)
            image_qualities.append(image_quality)
            timings_list.append(timings)

        fast_faces_list = self.fast_detector.detect_batch(images)

        for image, image_quality, fast_faces, timings in zip(
            images, image_qualities, fast_faces_list, timings_list
        ):
            if not image_quality.passed:
                timings["total_pipeline_ms"] = sum(timings.values())
                results.append(
                    {
                        "status": "quality_reject",
                        "quality_reason": image_quality.reason,
                        "quality_details": image_quality.details,
                        "timings": timings,
                    }
                )
                continue

            results.append(
                self._prepare_face_from_detection(image, image_quality, fast_faces, timings)
            )

        return results

    def prepare_face_inputs_from_images(self, images: list[np.ndarray]) -> list[Dict[str, Any]]:
        self._init()

        assert self.fast_detector is not None, "fast_detector not initialized"

        image_qualities: list[Any] = []
        timings_list: list[Dict[str, float]] = []
        processed_images: list[np.ndarray] = []
        results: list[Dict[str, Any]] = []

        for image in images:
            timings: Dict[str, float] = {}
            t0 = time.time()
            processed_image = self.preprocessor.process_image(image)
            timings["preprocess_ms"] = (time.time() - t0) * 1000

            t0 = time.time()
            image_quality = self.quality_gate.evaluate_image(processed_image)
            timings["quality_gate_pre_ms"] = (time.time() - t0) * 1000

            processed_images.append(processed_image)
            image_qualities.append(image_quality)
            timings_list.append(timings)

        fast_faces_list = self.fast_detector.detect_batch(processed_images)

        for image, image_quality, fast_faces, timings in zip(
            processed_images, image_qualities, fast_faces_list, timings_list
        ):
            if not image_quality.passed:
                timings["total_pipeline_ms"] = sum(timings.values())
                results.append(
                    {
                        "status": "quality_reject",
                        "quality_reason": image_quality.reason,
                        "quality_details": image_quality.details,
                        "timings": timings,
                    }
                )
                continue

            results.append(
                self._prepare_face_from_detection(image, image_quality, fast_faces, timings)
            )

        return results
    def process(self, image_bytes: bytes) -> Dict[str, Any]:
        prepared = self.prepare_face_input(image_bytes)
        if prepared["status"] != "ok":
            return prepared

        assert self.encoder is not None, "encoder not initialized"

        face_input = prepared.pop("face_input")
        timings = prepared["timings"]
        bbox_source = prepared["bbox_source"]
        bbox = prepared["bbox"]
        landmarks = prepared.get("landmarks")
        liveness_passed = prepared.get("liveness_passed")
        liveness_score = prepared.get("liveness_score")
        quality_details = prepared.get("quality_details")

        t0 = time.time()
        embeddings = self.encoder.encode_batch([face_input])
        if len(embeddings) == 0:
            raise RuntimeError("Batch encoder returned no embeddings")
        embedding = embeddings[0]
        timings["encode_ms"] = (time.time() - t0) * 1000

        if "liveness_ms" in timings and timings["liveness_ms"] > timings["encode_ms"]:
            logger.warning(
                "liveness_slower_than_encode liveness_ms=%.3f encode_ms=%.3f bbox_source=%s",
                timings["liveness_ms"],
                timings["encode_ms"],
                bbox_source,
            )
        if timings.get("liveness_ms", 0.0) > 50:
            logger.warning("slow_liveness_ms=%.2f", timings["liveness_ms"])

        timings["total_pipeline_ms"] = sum(timings.values())
        logger.info("bbox_source=%s", bbox_source)

        return {
            "status": "ok",
            "embedding": embedding,
            "bbox": bbox,
            "landmarks": landmarks,
            "liveness_passed": liveness_passed,
            "liveness_score": liveness_score,
            "bbox_source": bbox_source,
            "quality_details": quality_details,
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
