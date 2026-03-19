# faceid-core\app\ml\batch_encoder.py

import threading
import time
import numpy as np
from typing import List

from app.ml.embedding.arcface_encoder import ArcFaceEncoder


class BatchEncoder:

    def __init__(self, encoder, batch_size: int = 8, timeout: float = 0.01):
        self.encoder = encoder

        self.batch_size = batch_size
        self.timeout = timeout

        self.queue = []
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def encode(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding using L2 norm.
        """
        result = {}

        with self.condition:
            self.queue.append((embedding, result))
            self.condition.notify()

        # busy wait (короткий, допустимо)
        while "embedding" not in result:
            time.sleep(0.001)

        return result["embedding"]

    def _worker(self):
        while True:
            with self.condition:
                if not self.queue:
                    self.condition.wait(timeout=self.timeout)

                batch = self.queue[:self.batch_size]
                self.queue = self.queue[self.batch_size:]

            if not batch:
                continue

            embeddings = [item[0] for item in batch]
            results = [item[1] for item in batch]

            try:
                # Normalize each embedding
                normalized = [
                    self.encoder.normalize(emb)
                    for emb in embeddings
                ]

                for emb, res in zip(normalized, results):
                    res["embedding"] = emb

            except Exception as e:
                for res in results:
                    res["error"] = e