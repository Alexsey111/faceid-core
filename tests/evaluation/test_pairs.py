# tests/evaluation/test_pairs.py — unit-тесты evaluation.pairs (pure numpy, no app).

import numpy as np
import pytest

from evaluation.pairs import (
    DEFAULT_IMPOSTOR_RATIO,
    build_pairs_1to1,
    gallery_probe_split,
    pair_scores,
)

pytestmark = pytest.mark.unit


# ---------------- build_pairs_1to1 -------------------------------------------

def test_pairs_determinism_by_seed():
    ids = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    r1 = build_pairs_1to1(ids, impostor_ratio=5, seed=7)
    r2 = build_pairs_1to1(ids, impostor_ratio=5, seed=7)
    assert np.array_equal(r1[0], r2[0])
    assert np.array_equal(r1[1], r2[1])
    assert np.array_equal(r1[2], r2[2])


def test_pairs_different_seed_changes_impostor():
    ids = np.array(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"])
    r1 = build_pairs_1to1(ids, impostor_ratio=5, seed=1)
    r2 = build_pairs_1to1(ids, impostor_ratio=5, seed=2)
    # genuine часть одинакова (детерминирована структурой), impostor — разный.
    n_g = int(r1[2].sum())
    assert np.array_equal(r1[0][:n_g], r2[0][:n_g])
    assert not np.array_equal(r1[0][n_g:], r2[0][n_g:])


def test_pairs_genuine_all_same_id():
    # 3 фото одного id → C(3,2)=3 genuine пары, все label=1.
    ids = np.array(["x", "x", "x"])
    idx_i, idx_j, labels = build_pairs_1to1(ids, impostor_ratio=10, seed=1)
    assert len(labels) == 3
    assert labels.sum() == 3
    # все пары из индексов {0,1,2}
    pairs = set(zip(idx_i.tolist(), idx_j.tolist()))
    assert pairs == {(0, 1), (0, 2), (1, 2)}


def test_pairs_impostor_diff_id():
    ids = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    idx_i, idx_j, labels = build_pairs_1to1(ids, impostor_ratio=10, seed=42)
    # Все impostor-пары (label=0) — между разными id.
    impostor_mask = labels == 0
    for a, b in zip(idx_i[impostor_mask], idx_j[impostor_mask]):
        assert ids[a] != ids[b]
    # Все genuine-пары (label=1) — одинаковый id.
    genuine_mask = labels == 1
    for a, b in zip(idx_i[genuine_mask], idx_j[genuine_mask]):
        assert ids[a] == ids[b]


def test_pairs_balance_impostor_ratio():
    # 5 id × 2 фото → 5 genuine; всего diff-id пар = C(10,2)-5 = 40.
    # ratio=5 → 25 impostor запрошено, 40 доступно → ровно 25 (без дедуп-капа).
    ids = np.array(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"])
    idx_i, idx_j, labels = build_pairs_1to1(ids, impostor_ratio=5, seed=42)
    n_genuine = int((labels == 1).sum())
    n_impostor = int((labels == 0).sum())
    assert n_genuine == 5
    assert n_impostor == 25  # ровно 5× (датасет достаточно велик для дедуп-кап)


def test_pairs_unique_pairs():
    ids = np.repeat(np.arange(20), 3)  # 20 id × 3 фото
    idx_i, idx_j, labels = build_pairs_1to1(ids, impostor_ratio=5, seed=11)
    keys = set()
    for a, b, lab in zip(idx_i, idx_j, labels):
        key = (min(int(a), int(b)), max(int(a), int(b)), int(lab))
        assert key not in keys, f"duplicate pair {key}"
        keys.add(key)


def test_pairs_empty():
    idx_i, idx_j, labels = build_pairs_1to1(np.array([]), seed=1)
    assert len(idx_i) == 0 and len(idx_j) == 0 and len(labels) == 0


def test_pairs_singletons_no_genuine():
    # Все id уникальны → genuine нет; impostor по ratio×0=0 → пусто (но tries даёт 0).
    ids = np.array(["a", "b", "c"])
    idx_i, idx_j, labels = build_pairs_1to1(ids, impostor_ratio=10, seed=1)
    assert (labels == 1).sum() == 0
    # n_impostor = 10 * 0 = 0 → цикл не добавит ничего (условие len<0 ложно).
    assert (labels == 0).sum() == 0


# ---------------- gallery_probe_split ----------------------------------------

def test_gallery_probe_one_per_id_in_gallery():
    ids = np.array(["a", "a", "b", "c", "c", "c"])
    gallery, probe = gallery_probe_split(ids)
    # gallery содержит ровно 1 индекс на каждый уникальный id.
    assert len(gallery) == 3
    assert len(np.unique(ids[gallery])) == 3
    # probe — остальные, с тем же id, что есть в gallery.
    assert len(probe) == 3
    assert set(ids[gallery].tolist()) >= set(ids[probe].tolist())


def test_gallery_probe_single_photo_gallery_only():
    # id с 1 фото → только gallery, не queried.
    ids = np.array(["a", "b", "b", "c", "c", "c"])
    gallery, probe = gallery_probe_split(ids)
    # 'a' имеет 1 фото → в gallery, в probe его нет.
    a_idx = int(np.where(ids == "a")[0][0])
    assert a_idx in gallery.tolist()
    assert a_idx not in probe.tolist()


def test_gallery_probe_disjoint_complete():
    rng = np.random.default_rng(5)
    ids = rng.integers(0, 8, 30)
    gallery, probe = gallery_probe_split(ids)
    union = np.concatenate([gallery, probe])
    assert len(union) == len(ids)
    assert len(np.intersect1d(gallery, probe)) == 0
    assert np.array_equal(np.sort(union), np.arange(len(ids)))


def test_gallery_probe_deterministic_order():
    ids = np.array(["b", "a", "a", "b"])
    g1, p1 = gallery_probe_split(ids)
    g2, p2 = gallery_probe_split(ids)
    assert np.array_equal(g1, g2) and np.array_equal(p1, p2)


# ---------------- pair_scores ------------------------------------------------

def test_pair_scores_cosine_identical_is_one():
    E = np.eye(3, dtype=np.float32)  # L2-норм орты
    idx_i = np.array([0, 1, 2], dtype=np.int64)
    idx_j = np.array([0, 1, 2], dtype=np.int64)
    s = pair_scores(E, idx_i, idx_j)
    assert np.allclose(s, [1.0, 1.0, 1.0])


def test_pair_scores_orthogonal_is_zero():
    E = np.eye(3, dtype=np.float32)
    s = pair_scores(E, np.array([0, 1]), np.array([1, 2]))
    assert np.allclose(s, [0.0, 0.0])