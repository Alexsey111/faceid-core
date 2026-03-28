from __future__ import annotations

from pathlib import Path
from typing import Sequence
import logging
import time

import cv2
import numpy as np
import onnxruntime as ort

from app.core.config import settings


logger = logging.getLogger(__name__)


class OnnxArcFaceEncoder:
    """
    CPU-optimized ONNX ArcFace encoder.

    Ожидает crop лица 112x112 BGR uint8.
    Возвращает embedding float32.
    """

    INPUT_SIZE = (112, 112)

    def __init__(self, model_path: str | Path):
        model_path = str(model_path)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Для CPU single-request path обычно лучше фиксировать intra/inter threads.
        # Значения 1/1 или 2/1 часто лучше дефолта; зависит от машины.
        so.intra_op_num_threads = max(1, int(settings.ONNX_INTRA_OP_THREADS))
        so.inter_op_num_threads = max(1, int(settings.ONNX_INTER_OP_THREADS))

        # Уменьшаем шум логов ORT
        so.log_severity_level = 3

        self.session = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(
            "ArcFace ONNX loaded input=%s output=%s intra=%s inter=%s",
            self.input_name,
            self.output_name,
            so.intra_op_num_threads,
            so.inter_op_num_threads,
        )

    def _preprocess_single(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Вход:
            BGR uint8, желательно уже 112x112
        Выход:
            NCHW float32, shape=(1, 3, 112, 112)
        """
        if face_crop.size == 0:
            raise ValueError("Empty face crop")

        # Не делаем resize, если уже нужный размер
        if face_crop.shape[:2] != self.INPUT_SIZE:
            face_crop = cv2.resize(face_crop, self.INPUT_SIZE)

        # One-shot blob creation avoids extra cvtColor/transpose/astype passes.
        return cv2.dnn.blobFromImage(
            face_crop,
            scalefactor=1.0 / 128.0,
            size=self.INPUT_SIZE,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )

    def _preprocess_batch(self, face_crops: Sequence[np.ndarray]) -> np.ndarray:
        if not face_crops:
            return np.empty((0, 3, self.INPUT_SIZE[1], self.INPUT_SIZE[0]), dtype=np.float32)

        resized_crops = []
        for face_crop in face_crops:
            if face_crop.size == 0:
                raise ValueError("Empty face crop")
            if face_crop.shape[:2] != self.INPUT_SIZE:
                face_crop = cv2.resize(face_crop, self.INPUT_SIZE)
            resized_crops.append(face_crop)

        return cv2.dnn.blobFromImages(
            resized_crops,
            scalefactor=1.0 / 128.0,
            size=self.INPUT_SIZE,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )

    def _l2_normalize(self, embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding.astype(np.float32, copy=False)
        return (embedding / norm).astype(np.float32, copy=False)

    def _encode_single(self, face_crop: np.ndarray) -> np.ndarray:
        t0 = time.time()
        input_tensor = self._preprocess_single(face_crop)
        prep_ms = (time.time() - t0) * 1000.0

        t0 = time.time()
        output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        ort_ms = (time.time() - t0) * 1000.0

        logger.info("encode_prep_ms=%.3f ort_run_ms=%.3f", prep_ms, ort_ms)

        embedding = output[0]
        return self._l2_normalize(embedding)

    def encode(self, face_crop: np.ndarray) -> np.ndarray:
        return self._encode_single(face_crop)

    def encode_batch(self, face_crops: Sequence[np.ndarray]) -> np.ndarray:
        if not face_crops:
            return np.empty((0, 512), dtype=np.float32)

        input_tensor = self._preprocess_batch(face_crops)
        output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]

        embeddings = []
        for emb in output:
            embeddings.append(self._l2_normalize(emb))

        return np.asarray(embeddings, dtype=np.float32)
