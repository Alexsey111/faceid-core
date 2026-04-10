# faceid-core\app\ml\embedding\onnx_arcface_encoder.py

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
        so.intra_op_num_threads = max(1, int(settings.ONNX_INTRA_OP_THREADS))
        so.inter_op_num_threads = max(1, int(settings.ONNX_INTER_OP_THREADS))
        so.log_severity_level = 3

        self.session = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.last_single_timings: dict[str, float] = {}
        self.last_batch_timings: dict[str, float | int] = {}

        logger.info(
            "ArcFace ONNX loaded input=%s output=%s intra=%s inter=%s",
            self.input_name,
            self.output_name,
            so.intra_op_num_threads,
            so.inter_op_num_threads,
        )

    def _preprocess_single(self, face_crop: np.ndarray) -> np.ndarray:
        if face_crop.size == 0:
            raise ValueError("Empty face crop")

        if face_crop.shape[:2] != self.INPUT_SIZE:
            face_crop = cv2.resize(face_crop, self.INPUT_SIZE)

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
        t0 = time.perf_counter()
        input_tensor = self._preprocess_single(face_crop)
        encode_preprocess_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        encode_ort_run_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        embedding = self._l2_normalize(output[0])
        encode_postprocess_ms = (time.perf_counter() - t0) * 1000.0

        encode_ms = encode_preprocess_ms + encode_ort_run_ms + encode_postprocess_ms
        self.last_single_timings = {
            "encode_preprocess_ms": encode_preprocess_ms,
            "encode_ort_run_ms": encode_ort_run_ms,
            "encode_postprocess_ms": encode_postprocess_ms,
            "encode_ms": encode_ms,
        }

        logger.info(
            "encode_single preprocess_ms=%.3f ort_run_ms=%.3f postprocess_ms=%.3f total_ms=%.3f",
            encode_preprocess_ms,
            encode_ort_run_ms,
            encode_postprocess_ms,
            encode_ms,
        )

        return embedding

    def encode(self, face_crop: np.ndarray) -> np.ndarray:
        return self._encode_single(face_crop)

    def encode_batch(self, face_crops: Sequence[np.ndarray]) -> np.ndarray:
        if not face_crops:
            self.last_batch_timings = {
                "encode_preprocess_ms_total": 0.0,
                "encode_ort_run_ms_total": 0.0,
                "encode_postprocess_ms_total": 0.0,
                "encode_ms_total": 0.0,
                "encode_preprocess_ms_per_image": 0.0,
                "encode_ort_run_ms_per_image": 0.0,
                "encode_postprocess_ms_per_image": 0.0,
                "encode_ms_per_image": 0.0,
                "encode_batch_size": 0,
            }
            return np.empty((0, 512), dtype=np.float32)

        t0 = time.perf_counter()
        input_tensor = self._preprocess_batch(face_crops)
        encode_preprocess_ms_total = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        encode_ort_run_ms_total = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        embeddings = []
        for emb in output:
            embeddings.append(self._l2_normalize(emb))
        encode_postprocess_ms_total = (time.perf_counter() - t0) * 1000.0

        encoded = np.asarray(embeddings, dtype=np.float32)
        batch_size = len(face_crops)
        encode_ms_total = (
            encode_preprocess_ms_total + encode_ort_run_ms_total + encode_postprocess_ms_total
        )

        self.last_batch_timings = {
            "encode_preprocess_ms_total": encode_preprocess_ms_total,
            "encode_ort_run_ms_total": encode_ort_run_ms_total,
            "encode_postprocess_ms_total": encode_postprocess_ms_total,
            "encode_ms_total": encode_ms_total,
            "encode_preprocess_ms_per_image": encode_preprocess_ms_total / batch_size,
            "encode_ort_run_ms_per_image": encode_ort_run_ms_total / batch_size,
            "encode_postprocess_ms_per_image": encode_postprocess_ms_total / batch_size,
            "encode_ms_per_image": encode_ms_total / batch_size,
            "encode_batch_size": batch_size,
        }

        logger.info(
            "encode_batch size=%s preprocess_ms=%.3f ort_run_ms=%.3f postprocess_ms=%.3f total_ms=%.3f",
            batch_size,
            encode_preprocess_ms_total,
            encode_ort_run_ms_total,
            encode_postprocess_ms_total,
            encode_ms_total,
        )

        return encoded