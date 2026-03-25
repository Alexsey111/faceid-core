import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
from typing import Sequence

from app.core.config import settings


class OnnxArcFaceEncoder:

    def __init__(self):
        model_path = Path(settings.MODELS_DIR) / "buffalo_l" / "w600k_r50.onnx"

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
        return self.preprocess_batch([face_crop])[0:1]

    def preprocess_batch(self, face_crops: Sequence[np.ndarray]) -> np.ndarray:
        batch = np.empty((len(face_crops), 3, 112, 112), dtype=np.float32)
        for idx, face_crop in enumerate(face_crops):
            img = cv2.resize(face_crop, (112, 112))
            img = img.astype("float32")
            img = (img - 127.5) / 128.0
            batch[idx] = np.transpose(img, (2, 0, 1))
        return batch

    def normalize_batch(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return embeddings / norms

    def _encode_single(self, face_crop: np.ndarray) -> np.ndarray:
        input_tensor = self.preprocess(face_crop)
        embedding = self.session.run(None, {
            self.input_name: input_tensor
        })[0][0]
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm != 0:
            embedding = embedding / norm
        return embedding

    def encode_batch(self, face_crops: Sequence[np.ndarray]) -> np.ndarray:
        if not face_crops:
            return np.empty((0, 0), dtype=np.float32)

        input_tensor = self.preprocess_batch(face_crops)

        try:
            embeddings = self.session.run(None, {
                self.input_name: input_tensor
            })[0]
        except Exception:
            embeddings = np.stack([self._encode_single(face_crop) for face_crop in face_crops], axis=0)
            return embeddings

        embeddings = np.asarray(embeddings, dtype=np.float32)
        return self.normalize_batch(embeddings)

    def encode(self, face_crop: np.ndarray) -> np.ndarray:
        return self._encode_single(face_crop)
