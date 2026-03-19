# antispoof_model.py - Модель антиспуфинга

# app/ml/liveness/antispoof_model.py

from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort


class AntiSpoofModel:

    def __init__(self, model_path: str):

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Подготовка изображения для модели
        """

        img = cv2.resize(image, (80, 80))
        img = img.astype("float32") / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img

    def predict(self, face_crop: np.ndarray) -> float:
        """
        Возвращает вероятность того,
        что изображение является живым
        """

        input_tensor = self.preprocess(face_crop)

        output = self.session.run(None, {
            self.input_name: input_tensor
        })

        score = float(output[0][0][1])

        return score