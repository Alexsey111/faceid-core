# app\ml\batch_encoder.py

from __future__ import annotations

from collections import deque
import logging
import os
import threading
import time
from typing import Any

import numpy as np

from app.core.config import settings


logger = logging.getLogger(__name__)


class BatchEncoder:
    """
    Micro-batches face crops into a single encoder call.

    The wrapped encoder should expose `encode_batch(face_crops)` and `encode(face_crop)`.
    """

    def __init__(
        self,
        encoder: Any,
        batch_size: int = 8,
        timeout: float = 0.01,
        max_wait_guard_ms: float = 50.0,
    ):
        self.encoder = encoder
        self.batch_size = max(1, int(batch_size))
        self.timeout = max(0.0, float(timeout))
        self.max_wait_guard_ms = max(0.0, float(max_wait_guard_ms))

        self._queue: deque[tuple[np.ndarray, dict[str, Any], threading.Event, float]] = deque()
        self._condition = threading.Condition()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _encode_single_direct(self, face_crop: np.ndarray) -> np.ndarray:
        if hasattr(self.encoder, "_encode_single"):
            return self.encoder._encode_single(face_crop)
        return self.encoder.encode(face_crop)

    def encode(self, face_crop: np.ndarray) -> np.ndarray:
        result: dict[str, Any] = {}
        ready = threading.Event()

        with self._condition:
            queue_is_small = len(self._queue) <= 1

        if queue_is_small:
            return self._encode_single_direct(face_crop)

        with self._condition:
            self._queue.append((face_crop, result, ready, time.monotonic()))
            self._condition.notify()

        ready.wait()

        if "error" in result:
            raise result["error"]

        return result["embedding"]

    def encode_batch(self, face_crops: list[np.ndarray]) -> np.ndarray:
        return self.encoder.encode_batch(face_crops)

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while True:
                    if not self._queue:
                        self._condition.wait(timeout=self.timeout)
                        continue

                    oldest_wait_ms = (time.monotonic() - self._queue[0][3]) * 1000.0
                    if len(self._queue) >= self.batch_size or oldest_wait_ms >= self.max_wait_guard_ms:
                        batch: list[tuple[np.ndarray, dict[str, Any], threading.Event, float]] = []
                        while self._queue and len(batch) < self.batch_size:
                            batch.append(self._queue.popleft())
                        break

                    wait_s = min(self.timeout, max(0.0, (self.max_wait_guard_ms - oldest_wait_ms) / 1000.0))
                    self._condition.wait(timeout=wait_s)

            if not batch:
                continue

            face_crops = [item[0] for item in batch]
            results = [item[1] for item in batch]
            events = [item[2] for item in batch]

            try:
                logger.info("PID=%s batch_size=%d", os.getpid(), len(batch))
                t0 = time.time()
                embeddings = self.encoder.encode_batch(face_crops)
                logger.info("encode_batch_ms=%.3f", (time.time() - t0) * 1000.0)
                if len(embeddings) != len(batch):
                    raise RuntimeError("Batch encoder returned unexpected batch size")

                for embedding, result, ready in zip(embeddings, results, events):
                    result["embedding"] = embedding
                    ready.set()
            except Exception as exc:
                for result, ready in zip(results, events):
                    result["error"] = exc
                    ready.set()
