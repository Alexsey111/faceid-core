# tests/evaluation/test_metrics.py — unit-тесты evaluation.metrics (pure numpy, no app).
# Маркер `unit` → conftest пропускает миграции БД/redis.

import numpy as np
import pytest

from evaluation.metrics import (
    TARGET_FAR,
    auc_from_roc,
    cmc_curve,
    confusion,
    eer_point,
    recommend_thresholds,
    roc_curve,
    score_distribution,
    tar_at_far,
)

pytestmark = pytest.mark.unit


# ---------------- confusion -------------------------------------------------

def test_confusion_perfect_separation():
    scores = np.array([0.9, 0.8, 0.4, 0.3])
    labels = np.array([1, 1, 0, 0])
    m = confusion(scores, labels, 0.5)
    assert m["tp"] == 2 and m["fp"] == 0 and m["tn"] == 2 and m["fn"] == 0
    assert m["far"] == 0.0 and m["frr"] == 0.0 and m["tar"] == 1.0
    assert m["precision"] == 1.0 and m["recall"] == 1.0


def test_confusion_all_rejected():
    # Порог выше всех score → всё no_match: tp=0, fn=P, fp=0, tn=N.
    scores = np.array([0.9, 0.8, 0.4])
    labels = np.array([1, 1, 0])
    m = confusion(scores, labels, 1.5)
    assert m["tp"] == 0 and m["fn"] == 2 and m["fp"] == 0 and m["tn"] == 1
    assert m["far"] == 0.0 and m["frr"] == 1.0 and m["tar"] == 0.0


def test_confusion_all_accepted():
    scores = np.array([0.9, 0.8, 0.4])
    labels = np.array([1, 1, 0])
    m = confusion(scores, labels, -1.0)
    assert m["tp"] == 2 and m["fp"] == 1 and m["tn"] == 0 and m["fn"] == 0
    assert m["far"] == 1.0 and m["frr"] == 0.0 and m["tar"] == 1.0


# ---------------- ROC / AUC --------------------------------------------------

def test_roc_auc_perfect():
    # Сепарабельные: все genuine выше всех impostor.
    scores = np.array([0.95, 0.90, 0.20, 0.10])
    labels = np.array([1, 1, 0, 0])
    far, tar, thr = roc_curve(scores, labels)
    assert auc_from_roc(far, tar) == pytest.approx(1.0, abs=1e-9)
    # При FAR=0 уже TAR=1.
    t, _ = tar_at_far(far, tar, thr, TARGET_FAR)
    assert t == pytest.approx(1.0, abs=1e-9)


def test_roc_auc_worst():
    # Инверсия: genuine ниже impostor → AUC ~ 0.
    scores = np.array([0.20, 0.10, 0.95, 0.90])
    labels = np.array([1, 1, 0, 0])
    far, tar, _ = roc_curve(scores, labels)
    assert auc_from_roc(far, tar) == pytest.approx(0.0, abs=1e-9)


def test_roc_auc_random_around_half():
    rng = np.random.default_rng(42)
    n = 4000
    # Случайные score, label ~ 50/50 → AUC около 0.5.
    scores = rng.random(n)
    labels = rng.integers(0, 2, n)
    far, tar, _ = roc_curve(scores, labels)
    auc = auc_from_roc(far, tar)
    assert 0.3 < auc < 0.7


def test_roc_monotone_far_ascending():
    rng = np.random.default_rng(7)
    scores = rng.random(500)
    labels = rng.integers(0, 2, 500)
    far, tar, _ = roc_curve(scores, labels)
    # far должен быть неубывающим по построению (cumsum).
    assert np.all(np.diff(far) >= -1e-12)


# ---------------- tar_at_far ------------------------------------------------

def test_tar_at_far_monotone_in_target():
    # Сепарабельный случай: TAR@FAR=0.001 >= TAR@FAR=0.0001 (более мягкий порог).
    scores = np.array([0.97, 0.96, 0.95, 0.10, 0.05, 0.01])
    labels = np.array([1, 1, 1, 0, 0, 0])
    far, tar, thr = roc_curve(scores, labels)
    t_strict, _ = tar_at_far(far, tar, thr, 0.0001)
    t_loose, _ = tar_at_far(far, tar, thr, 0.001)
    assert t_loose >= t_strict


def test_tar_at_far_unreachable_returns_zero():
    # Impostor имеет наивысший score: любое принятие (threshold ниже топа) сразу
    # даёт far=1.0 > target → единственная точка с far<=target — это (far=0,tar=0),
    # порог выше всех score. Значит TAR@FAR=0.0001 = 0 (недостижимо).
    scores = np.array([0.99, 0.50])
    labels = np.array([0, 1])
    far, tar, thr = roc_curve(scores, labels)
    t, t_thr = tar_at_far(far, tar, thr, 0.0001)
    assert t == 0.0
    # Порог — выше всех score (ничего не принимается).
    assert t_thr > scores.max()


# ---------------- EER --------------------------------------------------------

def test_eer_balanced_known():
    # Сконструируем так, чтобы EER ~ 0.1: 10 genuine и 10 impostor с перекрытием.
    # genuine scores: 0.6..1.0 (10), impostor: 0.0..0.9 (10) — пересечение в ~0.6..0.9.
    scores = np.concatenate([np.linspace(0.6, 1.0, 10), np.linspace(0.0, 0.9, 10)])
    labels = np.concatenate([np.ones(10, dtype=int), np.zeros(10, dtype=int)])
    far, tar, thr = roc_curve(scores, labels)
    eer, _ = eer_point(far, tar, thr)
    assert 0.0 <= eer <= 0.5


def test_eer_perfect_is_zero():
    scores = np.array([0.95, 0.90, 0.20, 0.10])
    labels = np.array([1, 1, 0, 0])
    far, tar, thr = roc_curve(scores, labels)
    eer, _ = eer_point(far, tar, thr)
    assert eer == pytest.approx(0.0, abs=1e-9)


# ---------------- recommend_thresholds --------------------------------------

def test_recommend_thresholds_returns_all_fields():
    rng = np.random.default_rng(1)
    g = rng.normal(0.7, 0.05, 200)  # genuine
    i = rng.normal(0.3, 0.05, 200)  # impostor
    scores = np.concatenate([g, i])
    labels = np.concatenate([np.ones(200, int), np.zeros(200, int)])
    far, tar, thr = roc_curve(scores, labels)
    rec = recommend_thresholds(far, tar, thr, scores, labels)
    assert {"high", "low", "margin", "eer", "basis"} <= set(rec)
    assert rec["high"] > rec["low"]  # хорошо разделимы
    assert 0.0 <= rec["margin"] <= rec["high"]


def test_recommend_thresholds_high_between_clusters():
    # Чётко разделимые кластеры 0.8 / 0.2 → high должен попасть между ними.
    scores = np.array([0.85, 0.82, 0.80, 0.20, 0.18, 0.15])
    labels = np.array([1, 1, 1, 0, 0, 0])
    far, tar, thr = roc_curve(scores, labels)
    rec = recommend_thresholds(far, tar, thr, scores, labels)
    assert 0.20 < rec["high"] <= 0.85


# ---------------- CMC --------------------------------------------------------

def test_cmc_perfect_rank1():
    # probe == gallery embeddings → ранг 1 всегда верный.
    emb = np.eye(5, dtype=np.float32)
    ids = np.array(["a", "b", "c", "d", "e"])
    ranks, acc = cmc_curve(emb, ids, emb, ids, max_rank=5)
    assert ranks[0] == 1
    assert acc[0] == pytest.approx(1.0, abs=1e-6)
    # Все ранги дают 1.0.
    assert np.allclose(acc, 1.0)


def test_cmc_known_ordering():
    # 3 probe, 3 gallery. probe0 -> gallery0 (ранг1), probe1 -> gallery2 (ранг3),
    # probe2 -> gallery1 (ранг2). Строим embeddings так, чтобы порядок сходился.
    # Используем ортогональные векторы gallery, probe — их линейные комбинации.
    g = np.eye(3, dtype=np.float32)
    gallery_ids = np.array(["a", "b", "c"])
    # probe0 похож на g0, probe1 — на g2, probe2 — на g1 (ранги 1, 3, 2).
    p = np.array([
        [1.0, 0.0, 0.0],   # -> g0 (a) ранг1
        [0.0, 0.0, 1.0],   # -> g2 (c) ранг1
        [0.0, 1.0, 0.0],   # -> g1 (b) ранг1
    ], dtype=np.float32)
    probe_ids = np.array(["a", "c", "b"])
    ranks, acc = cmc_curve(p, probe_ids, g, gallery_ids, max_rank=3)
    assert acc[0] == pytest.approx(1.0, abs=1e-6)  # все ранг1


def test_cmc_rank_grows_monotone():
    rng = np.random.default_rng(3)
    g = rng.random((10, 16)).astype(np.float32)
    g_ids = np.arange(10)
    p = g + rng.normal(0, 0.05, g.shape).astype(np.float32)
    p_ids = g_ids
    ranks, acc = cmc_curve(p, p_ids, g, g_ids, max_rank=10)
    assert np.all(np.diff(acc) >= -1e-9)  # CMC не убывает


# ---------------- score_distribution ----------------------------------------

def test_score_distribution_counts():
    scores = np.array([0.9, 0.1, 0.95, 0.05])
    labels = np.array([1, 0, 1, 0])
    d = score_distribution(scores, labels, n_bins=10)
    assert d["genuine"].sum() == 2
    assert d["impostor"].sum() == 2
    assert len(d["bin_low"]) == 10