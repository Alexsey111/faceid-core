# tests/evaluation/test_extract.py — регрессия на баг многократного батча в _flush.
# Баг: ids/files не очищались после _flush → при N батчах _FLUSH_IDS рос геометрически
# (extend добавлял весь накопленный список заново на каждом батче). Smoke с 1 батчем
# баг не ловил. Тест с фейковым encoder'ом (без ONNX-моделей).

import numpy as np
import pytest

import evaluation.extract as ex
from evaluation.extract import _consume_flushed, _flush

pytestmark = pytest.mark.unit


class _FakeEncoder:
    """Возвращает батч эмбеддингов той же длины, что и кропы (L2-норм единицы)."""

    def __init__(self):
        self.batch_sizes: list[int] = []

    def encode_batch(self, crops):
        n = len(crops)
        self.batch_sizes.append(n)
        # единичные орты в (n, 4) — L2-нормированы
        e = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            e[i, i % 4] = 1.0
        return e


def test_flush_clears_buffers_no_geometric_growth():
    """После баг-фикса _flush очищает crops/ids/files → многократный батч не раздувает накопитель."""
    enc = _FakeEncoder()
    # 3 батча по 2 → должно дать 6 записей, а не 2+4+6=12 (геометрический баг).
    for batch_n in range(3):
        crops = [np.zeros((112, 112, 3), np.uint8) for _ in range(2)]
        ids = [f"id_{batch_n}_a", f"id_{batch_n}_b"]
        files = [f"f_{batch_n}_a", f"f_{batch_n}_b"]
        _flush(crops, ids, files, enc)
        # буферы должны быть очищены _flush
        assert len(crops) == 0 and len(ids) == 0 and len(files) == 0

    emb, ids_out, files_out = _consume_flushed()
    assert len(ids_out) == 6, f"expected 6, got {len(ids_out)} (geometric growth bug)"
    assert len(files_out) == 6
    assert emb.shape == (6, 4)
    assert enc.batch_sizes == [2, 2, 2]


def test_flush_partial_then_consume():
    enc = _FakeEncoder()
    crops = [np.zeros((112, 112, 3), np.uint8) for _ in range(5)]
    ids = list("abcde")
    files = ["f1", "f2", "f3", "f4", "f5"]
    _flush(crops, ids, files, enc)
    emb, ids_out, files_out = _consume_flushed()
    assert ids_out == list("abcde")
    assert files_out == ["f1", "f2", "f3", "f4", "f5"]
    assert emb.shape == (5, 4)
    # накопитель очищен после consume
    assert len(ex._FLUSH_EMB) == 0 and len(ex._FLUSH_IDS) == 0 and len(ex._FLUSH_FILES) == 0