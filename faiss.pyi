from __future__ import annotations
from typing import Tuple

import numpy as np


def normalize_L2(x: np.ndarray) -> None: ...


def omp_set_num_threads(n: int) -> None: ...


class Index:
    d: int
    ntotal: int

    def add(self, vectors: np.ndarray) -> None: ...

    def search(
        self,
        x: np.ndarray,
        k: int,
        distances=None,
        labels=None
    ) -> Tuple[np.ndarray, np.ndarray]: ...


class IndexFlatIP(Index):
    def __init__(self, dim: int) -> None: ...
