from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort


class OnnxLivenessChecker:
    """
    Passive liveness detection based on MiniFASNetV2 / Silent-Face anti-spoof ONNX.
    """

    def __init__(self, model_path: str, threshold: float = 0.5):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.threshold = float(threshold)

    def preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """
        Prepare the face image for the ONNX model.
        """
        img = cv2.resize(face_img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, face_img: np.ndarray) -> Tuple[bool, float]:
        """
        Returns:
        (is_live, confidence)
        """
        input_tensor = self.preprocess(face_img)

        outputs = self.session.run(None, {self.input_name: input_tensor})
        scores = np.asarray(outputs[0], dtype=np.float32)

        real_score = float(scores[0][0])
        spoof_score = float(scores[0][1])
        _ = spoof_score

        score = real_score
        is_live = score >= self.threshold

        return is_live, score
