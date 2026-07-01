# app\ml\detection\fast_detector.py

import logging
from time import perf_counter

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class FastFaceDetector:
    def __init__(self, model_path, config_path):
        self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        self.last_detect_timings: dict[str, float | bool | int] = {}
        self.last_batch_timings: dict[str, float | bool | int] = {}

    @staticmethod
    def _decode_detections(detections: np.ndarray, width: int, height: int):
        faces = []

        if detections.ndim != 4:
            return faces

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < 0.3:
                continue

            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            x1, y1, x2, y2 = box.astype(int)

            faces.append([x1, y1, x2, y2, confidence])

        return faces

    def detect(self, image: np.ndarray):
        h, w = image.shape[:2]

        t0 = perf_counter()
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        detect_blob_ms = (perf_counter() - t0) * 1000.0

        self.net.setInput(blob)

        t0 = perf_counter()
        detections = self.net.forward()
        detect_forward_ms = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        faces = self._decode_detections(np.asarray(detections), w, h)
        detect_decode_ms = (perf_counter() - t0) * 1000.0

        detect_ms = detect_blob_ms + detect_forward_ms + detect_decode_ms
        self.last_detect_timings = {
            "detect_blob_ms": detect_blob_ms,
            "detect_forward_ms": detect_forward_ms,
            "detect_decode_ms": detect_decode_ms,
            "detect_ms": detect_ms,
            "detect_batch_fallback": False,
            "detect_batch_size": 1,
        }

        return faces

    def detect_batch(self, images: list[np.ndarray]) -> list[list[list[float]]]:
        self.last_batch_timings = {
            "detect_blob_ms_total": 0.0,
            "detect_forward_ms_total": 0.0,
            "detect_decode_ms_total": 0.0,
            "detect_ms_total": 0.0,
            "detect_blob_ms_per_image": 0.0,
            "detect_forward_ms_per_image": 0.0,
            "detect_decode_ms_per_image": 0.0,
            "detect_ms_per_image": 0.0,
            "detect_batch_fallback": False,
            "detect_batch_size": len(images),
        }

        if not images:
            return []

        if len(images) == 1:
            faces = self.detect(images[0])
            single = self.last_detect_timings
            self.last_batch_timings = {
                "detect_blob_ms_total": float(single.get("detect_blob_ms", 0.0)),
                "detect_forward_ms_total": float(single.get("detect_forward_ms", 0.0)),
                "detect_decode_ms_total": float(single.get("detect_decode_ms", 0.0)),
                "detect_ms_total": float(single.get("detect_ms", 0.0)),
                "detect_blob_ms_per_image": float(single.get("detect_blob_ms", 0.0)),
                "detect_forward_ms_per_image": float(single.get("detect_forward_ms", 0.0)),
                "detect_decode_ms_per_image": float(single.get("detect_decode_ms", 0.0)),
                "detect_ms_per_image": float(single.get("detect_ms", 0.0)),
                "detect_batch_fallback": False,
                "detect_batch_size": 1,
            }
            return [faces]

        try:
            t0 = perf_counter()
            blob = cv2.dnn.blobFromImages(
                images,
                scalefactor=1.0,
                size=(300, 300),
                mean=(104.0, 177.0, 123.0),
                swapRB=False,
                crop=False,
            )
            detect_blob_ms_total = (perf_counter() - t0) * 1000.0

            self.net.setInput(blob)

            t0 = perf_counter()
            detections = np.asarray(self.net.forward())
            detect_forward_ms_total = (perf_counter() - t0) * 1000.0

            if detections.ndim == 4 and detections.shape[0] == len(images):
                t0 = perf_counter()
                results: list[list[list[float]]] = []
                for idx, image in enumerate(images):
                    h, w = image.shape[:2]
                    batch_detections = detections[idx : idx + 1]
                    results.append(self._decode_detections(batch_detections, w, h))
                detect_decode_ms_total = (perf_counter() - t0) * 1000.0

                detect_ms_total = (
                    detect_blob_ms_total + detect_forward_ms_total + detect_decode_ms_total
                )
                batch_size = len(images)

                self.last_batch_timings = {
                    "detect_blob_ms_total": detect_blob_ms_total,
                    "detect_forward_ms_total": detect_forward_ms_total,
                    "detect_decode_ms_total": detect_decode_ms_total,
                    "detect_ms_total": detect_ms_total,
                    "detect_blob_ms_per_image": detect_blob_ms_total / batch_size,
                    "detect_forward_ms_per_image": detect_forward_ms_total / batch_size,
                    "detect_decode_ms_per_image": detect_decode_ms_total / batch_size,
                    "detect_ms_per_image": detect_ms_total / batch_size,
                    "detect_batch_fallback": False,
                    "detect_batch_size": batch_size,
                }

                logger.info(
                    "detect_batch size=%s blob_ms=%.3f forward_ms=%.3f decode_ms=%.3f total_ms=%.3f fallback=%s",
                    batch_size,
                    detect_blob_ms_total,
                    detect_forward_ms_total,
                    detect_decode_ms_total,
                    detect_ms_total,
                    False,
                )
                return results

            logger.warning(
                "detect_batch_unexpected_shape shape=%s batch=%s -> serial_fallback",
                tuple(detections.shape),
                len(images),
            )

        except Exception as exc:
            logger.warning(
                "detect_batch_exception batch=%s error=%s -> serial_fallback",
                len(images),
                exc,
            )

        results = []
        detect_blob_ms_total = 0.0
        detect_forward_ms_total = 0.0
        detect_decode_ms_total = 0.0

        for image in images:
            faces = self.detect(image)
            results.append(faces)
            single = self.last_detect_timings
            detect_blob_ms_total += float(single.get("detect_blob_ms", 0.0))
            detect_forward_ms_total += float(single.get("detect_forward_ms", 0.0))
            detect_decode_ms_total += float(single.get("detect_decode_ms", 0.0))

        batch_size = len(images)
        detect_ms_total = detect_blob_ms_total + detect_forward_ms_total + detect_decode_ms_total

        self.last_batch_timings = {
            "detect_blob_ms_total": detect_blob_ms_total,
            "detect_forward_ms_total": detect_forward_ms_total,
            "detect_decode_ms_total": detect_decode_ms_total,
            "detect_ms_total": detect_ms_total,
            "detect_blob_ms_per_image": detect_blob_ms_total / batch_size,
            "detect_forward_ms_per_image": detect_forward_ms_total / batch_size,
            "detect_decode_ms_per_image": detect_decode_ms_total / batch_size,
            "detect_ms_per_image": detect_ms_total / batch_size,
            "detect_batch_fallback": True,
            "detect_batch_size": batch_size,
        }

        logger.info(
            "detect_batch size=%s blob_ms=%.3f forward_ms=%.3f decode_ms=%.3f total_ms=%.3f fallback=%s",
            batch_size,
            detect_blob_ms_total,
            detect_forward_ms_total,
            detect_decode_ms_total,
            detect_ms_total,
            True,
        )
        return results