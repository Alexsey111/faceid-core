# app\services\faiss_index.py

import os
import pickle

import faiss
import numpy as np
from typing import List, Dict

from app.core.config import settings

class FaissIndex:
    """
    Минимальный FAISS индекс (in-memory).
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.index_path = "faiss.index"
        self.meta_path = "faiss_meta.pkl"
        self.index = faiss.IndexFlatIP(dim)  # cosine через inner product
        if hasattr(faiss, "omp_set_num_threads"):  # type: ignore[attr-defined]
            faiss.omp_set_num_threads(2)
        self.user_ids: List[int] = []
        self._seen = set()

        if getattr(settings, "FAISS_PERSIST_ENABLED", True):
            self._load()

    def add(self, vectors: np.ndarray, user_ids: List[int]):
        """
        vectors: shape (N, 512)
        """
        if len(vectors) == 0:
            return

        # normalize → cosine similarity
        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.user_ids.extend(user_ids)

        self._save()

    def add_one(self, vector: np.ndarray, user_id: int):
        """
        Добавить один embedding в индекс.
        """

        if vector is None or len(vector) == 0:
            return

        vector = vector.astype("float32").reshape(1, -1)
        faiss.normalize_L2(vector)
        key = (user_id, vector.tobytes())
        if key in self._seen:
            return

        self.index.add(vector)
        self.user_ids.append(user_id)
        self._seen.add(key)

        self._save()

    def reset(self):
        """
        Полная очистка индекса (для rebuild).
        """
        self.index = faiss.IndexFlatIP(self.dim)
        self.user_ids = []
        self._seen = set()

    def rebuild(self, items: List[Dict]):
        """
        Полная пересборка индекса из списка:
        [{user_id, embedding}]
        """

        if not items:
            self.reset()
            return

        vectors = []
        user_ids = []

        for item in items:
            vec = np.asarray(item["embedding"], dtype=np.float32)

            if vec.ndim != 1 or vec.shape[0] != self.dim:
                continue

            norm = np.linalg.norm(vec)
            if norm == 0.0:
                continue

            vec = vec / norm

            vectors.append(vec)
            user_ids.append(item["user_id"])

        if not vectors:
            self.reset()
            return

        matrix = np.vstack(vectors).astype("float32")

        self.reset()
        self.index.add(matrix)
        self.user_ids = user_ids
        self._seen = {
            (uid, matrix[i].tobytes())
            for i, uid in enumerate(user_ids)
        }

        self._save()

    def search(self, vector: np.ndarray, k: int = 2) -> List[Dict]:
        if self.index.ntotal == 0:
            return []

        vector = vector.astype("float32").reshape(1, -1)
        faiss.normalize_L2(vector)

        scores, indices = self.index.search(vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            results.append({
                "user_id": self.user_ids[idx],
                "similarity": float(score)
            })

        return results

    def _save(self):
        if not getattr(settings, "FAISS_PERSIST_ENABLED", True):
            return
        try:
            write_index = getattr(faiss, "write_index", None)
            if write_index is None:
                return

            write_index(self.index, self.index_path)
            with open(self.meta_path, "wb") as f:
                pickle.dump(self.user_ids, f)
        except Exception:
            pass

    def _load(self):
        if not getattr(settings, "FAISS_PERSIST_ENABLED", True):
            return
        if not os.path.exists(self.index_path):
            return

        try:
            read_index = getattr(faiss, "read_index", None)
            if read_index is None:
                return

            self.index = read_index(self.index_path)

            if os.path.exists(self.meta_path):
                with open(self.meta_path, "rb") as f:
                    self.user_ids = pickle.load(f)
        except Exception:
            # fallback → пустой индекс
            self.index = faiss.IndexFlatIP(self.dim)
            self.user_ids = []
