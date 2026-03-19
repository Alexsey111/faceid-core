import numpy as np


class ArcFaceEncoder:
    """
    L2 normalization for embeddings produced by InsightFace.
    """

    def normalize(self, embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Invalid embedding vector")
        return embedding / norm

