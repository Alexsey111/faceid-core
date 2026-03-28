# app/ml/detection/fast_detector.py

import cv2
import numpy as np


class FastFaceDetector:
    def __init__(self, model_path, config_path):
        self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)

    @staticmethod
    def _decode_detections(detections: np.ndarray, width: int, height: int):
        faces = []

        if detections.ndim != 4:
            return faces

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < 0.6:
                continue

            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            x1, y1, x2, y2 = box.astype(int)

            faces.append([x1, y1, x2, y2, confidence])

        return faces

    def detect(self, image: np.ndarray):
        h, w = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0)
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        return self._decode_detections(np.asarray(detections), w, h)

    def detect_batch(self, images: list[np.ndarray]) -> list[list[list[float]]]:
        if not images:
            return []
        if len(images) == 1:
            return [self.detect(images[0])]

        try:
            blob = cv2.dnn.blobFromImages(
                images,
                scalefactor=1.0,
                size=(300, 300),
                mean=(104.0, 177.0, 123.0),
                swapRB=False,
                crop=False,
            )
            self.net.setInput(blob)
            detections = np.asarray(self.net.forward())

            if detections.ndim == 4 and detections.shape[0] == len(images):
                results: list[list[list[float]]] = []
                for idx, image in enumerate(images):
                    h, w = image.shape[:2]
                    batch_detections = detections[idx : idx + 1]
                    results.append(self._decode_detections(batch_detections, w, h))
                return results

            if detections.ndim == 4 and detections.shape[0] == 1 and len(images) == 1:
                h, w = images[0].shape[:2]
                return [self._decode_detections(detections, w, h)]
        except Exception:
            pass

        return [self.detect(image) for image in images]
