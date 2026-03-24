import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path

from app.core.config import settings


class OnnxArcFaceEncoder:

    def __init__(self):
        model_path = Path(settings.MODELS_DIR) / "models" / "buffalo_l" / "w600k_r50.onnx"

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=so,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_crop, (112, 112))
        img = img.astype("float32")
        img = (img - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def encode(self, face_crop: np.ndarray) -> np.ndarray:
        input_tensor = self.preprocess(face_crop)

        embedding = self.session.run(None, {
            self.input_name: input_tensor
        })[0][0]

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm != 0:
            embedding = embedding / norm

        return embedding
