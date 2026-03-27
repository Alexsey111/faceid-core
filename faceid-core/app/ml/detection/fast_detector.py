# app/ml/detection/fast_detector.py

import cv2
import numpy as np


class FastFaceDetector:
    def __init__(self, model_path, config_path):
        self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)

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

        faces = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < 0.6:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            faces.append([x1, y1, x2, y2, confidence])

        return faces