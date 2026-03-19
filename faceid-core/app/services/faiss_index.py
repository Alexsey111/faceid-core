# faceid-core\app\services\faiss_index.py

import faiss
import numpy as np
from typing import List, Dict

class FaissIndex:
    """
    Минимальный FAISS индекс (in-memory).
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine через inner product
        if hasattr(faiss, "omp_set_num_threads"):  # type: ignore[attr-defined]
            faiss.omp_set_num_threads(2)
        self.user_ids: List[int] = []

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

    def add_one(self, vector: np.ndarray, user_id: int):
        """
        Добавить один embedding в индекс.
        """

        if vector is None or len(vector) == 0:
            return

        vector = vector.astype("float32").reshape(1, -1)
        faiss.normalize_L2(vector)

        self.index.add(vector)
        self.user_ids.append(user_id)

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
