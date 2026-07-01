# tests/evaluation/test_threshold.py — интеграционные unit-тесты 1:N (split + CMC)
# и связки threshold↔FAR. Pure numpy, без app/инфры.

import numpy as np
import pytest

from evaluation.metrics import cmc_curve, roc_curve, tar_at_far
from evaluation.pairs import gallery_probe_split

pytestmark = pytest.mark.unit


def test_cmc_rank1_one_when_probe_equals_gallery():
    # probe == gallery (орты) → ранг 1 всегда верный → rank-1 accuracy = 1.0.
    E = np.eye(6, dtype=np.float32)
    ids = np.array(["a", "b", "c", "d", "e", "f"])
    ranks, acc = cmc_curve(E, ids, E, ids, max_rank=6)
    assert ranks[0] == 1
    assert acc[0] == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(acc, 1.0)


def test_gallery_probe_split_feeds_cmc():
    # Строим gallery/probe из synthetic ids, затем CMC: rank-1 должен быть высоким,
    # если probe-эмбеддинги близки к своему gallery-эталону.
    rng = np.random.default_rng(0)
    n_ids = 8
    per_id = 3
    ids = np.repeat(np.arange(n_ids), per_id)
    # gallery-эталоны — случайные орты; probe — эталон + шум (тот же id).
    base = rng.standard_normal((n_ids, 16)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    # полный массив эмбеддингов: для каждого id — base[id] (+ шум для 2,3-го фото)
    E = np.zeros((n_ids * per_id, 16), dtype=np.float32)
    for i, gid in enumerate(ids):
        v = base[gid] + rng.normal(0, 0.02, 16).astype(np.float32)
        E[i] = v / np.linalg.norm(v)
    g_idx, p_idx = gallery_probe_split(ids)
    g_emb, g_ids = E[g_idx], ids[g_idx]
    p_emb, p_ids = E[p_idx], ids[p_idx]
    ranks, acc = cmc_curve(p_emb, p_ids, g_emb, g_ids, max_rank=n_ids)
    assert acc[0] == pytest.approx(1.0, abs=1e-6)  # rank-1 = 1.0 (шум мал)
    assert acc[-1] == pytest.approx(1.0, abs=1e-6)  # ранг = |gallery| → 1.0


def test_threshold_at_far_lookup_from_roc():
    # На сепарабельных данных TAR@FAR=0.001 = 1.0 (порог между кластерами).
    scores = np.array([0.95, 0.90, 0.20, 0.10])
    labels = np.array([1, 1, 0, 0])
    far, tar, thr = roc_curve(scores, labels)
    t, t_thr = tar_at_far(far, tar, thr, 0.001)
    assert t == pytest.approx(1.0, abs=1e-9)
    assert 0.20 < t_thr <= 0.95  # порог между impostor и genuine кластерами


def test_threshold_far_monotone_relaxing():
    # Более мягкий target_far даёт TAR >= строгого.
    scores = np.array([0.97, 0.96, 0.10, 0.05, 0.01])
    labels = np.array([1, 1, 0, 0, 0])
    far, tar, thr = roc_curve(scores, labels)
    t_strict, _ = tar_at_far(far, tar, thr, 0.0001)
    t_loose, _ = tar_at_far(far, tar, thr, 0.5)
    assert t_loose >= t_strict