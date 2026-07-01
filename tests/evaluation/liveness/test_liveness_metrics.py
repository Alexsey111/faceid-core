# tests/evaluation/liveness/test_liveness_metrics.py — unit-тесты liveness-метрик (pure numpy).

import numpy as np
import pytest

from evaluation.liveness.metrics import (
    ATTACK_TYPES,
    acer_overall,
    apcer_per_type,
    confusion_liveness,
    recommend_threshold_liveness,
    roc_liveness,
    score_distribution_liveness,
)

pytestmark = pytest.mark.unit


# ---------------- confusion_liveness ----------------------------------------

def test_confusion_perfect():
    # live: high scores, attack: low scores, порог 0.5 → идеальное разделение.
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    m = confusion_liveness(scores, labels, 0.5)
    assert m["tp"] == 2 and m["fp"] == 0 and m["tn"] == 2 and m["fn"] == 0
    assert m["apcer"] == 0.0 and m["npcer"] == 0.0
    assert m["accuracy"] == 1.0 and m["acer"] == 0.0


def test_confusion_all_rejected():
    # порог выше всех → всё spoof: tp=0, fn=2(live), fp=0, tn=2(attack).
    scores = np.array([0.9, 0.8, 0.2])
    labels = np.array([1, 1, 0])
    m = confusion_liveness(scores, labels, 1.5)
    assert m["tp"] == 0 and m["fn"] == 2 and m["fp"] == 0 and m["tn"] == 1
    assert m["npcer"] == 1.0  # все live отбракованы
    assert m["apcer"] == 0.0  # атак не пропущено


def test_confusion_all_accepted():
    scores = np.array([0.9, 0.8, 0.2])
    labels = np.array([1, 1, 0])
    m = confusion_liveness(scores, labels, -1.0)
    assert m["tp"] == 2 and m["fp"] == 1 and m["tn"] == 0 and m["fn"] == 0
    assert m["apcer"] == 1.0 and m["npcer"] == 0.0
    assert m["accuracy"] == pytest.approx(2 / 3)


def test_confusion_acer_formula():
    scores = np.array([0.6, 0.4, 0.55, 0.45])
    labels = np.array([1, 1, 0, 0])
    m = confusion_liveness(scores, labels, 0.5)
    # tp=1 (0.6), fn=1 (0.4), fp=1 (0.55), tn=1 (0.45)
    assert m["apcer"] == 0.5 and m["npcer"] == 0.5
    assert m["acer"] == 0.5


# ---------------- apcer_per_type ---------------------------------------------

def test_apcer_per_type_known():
    # 3 print (2 accepted), 2 replay (0 accepted), 1 cutout (1 accepted).
    scores = np.array([0.9, 0.7, 0.2, 0.1, 0.05, 0.8])
    labels = np.array([0, 0, 0, 0, 0, 0])
    types = np.array(["print", "print", "print", "replay", "replay", "cutout"])
    pt = apcer_per_type(scores, labels, types, 0.5)
    assert pt["print"] == {"n": 3, "accepted": 2, "apcer": pytest.approx(2 / 3)}
    assert pt["replay"] == {"n": 2, "accepted": 0, "apcer": 0.0}
    assert pt["cutout"] == {"n": 1, "accepted": 1, "apcer": 1.0}


def test_apcer_per_type_ignores_live():
    # live-семплы (label=1) не должны попасть в per-type APCER.
    scores = np.array([0.9, 0.2, 0.9])
    labels = np.array([1, 0, 0])
    types = np.array(["live", "print", "print"])
    pt = apcer_per_type(scores, labels, types, 0.5)
    assert pt["print"]["n"] == 2
    assert pt["print"]["accepted"] == 1
    assert pt["replay"]["n"] == 0  # нет replay-атак


def test_apcer_per_type_all_types_present():
    # все типы должны быть в результате (даже если n=0).
    scores = np.array([0.1, 0.9])
    labels = np.array([0, 0])
    types = np.array(["print", "print"])
    pt = apcer_per_type(scores, labels, types, 0.5)
    assert set(pt.keys()) == set(ATTACK_TYPES)


# ---------------- acer_overall / ROC -----------------------------------------

def test_acer_overall_formula():
    assert acer_overall(0.1, 0.2) == pytest.approx(0.15)
    assert acer_overall(0.0, 0.0) == 0.0


def test_roc_monotone_apcer_ascending():
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.normal(0.7, 0.1, 200), rng.normal(0.3, 0.1, 200)])
    labels = np.concatenate([np.ones(200, int), np.zeros(200, int)])
    apcer, tpr, thr = roc_liveness(scores, labels)
    assert np.all(np.diff(apcer) >= -1e-12)  # apcer не убывает
    assert apcer[0] == 0.0 and apcer[-1] == 1.0


def test_roc_auc_perfect():
    scores = np.array([0.95, 0.90, 0.20, 0.10])
    labels = np.array([1, 1, 0, 0])
    apcer, tpr, _ = roc_liveness(scores, labels)
    from evaluation.metrics import auc_from_roc
    assert auc_from_roc(apcer, tpr) == pytest.approx(1.0, abs=1e-9)


# ---------------- recommend_threshold ----------------------------------------

def test_recommend_threshold_minimizes_acer():
    # live ~0.8, attack ~0.2; оптимальный порог между ними → ACER≈0.
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0.8, 0.05, 100), rng.normal(0.2, 0.05, 300)])
    labels = np.concatenate([np.ones(100, int), np.zeros(300, int)])
    types = np.array(["live"] * 100 + ["print"] * 100 + ["replay"] * 100 + ["cutout"] * 100)
    rec = recommend_threshold_liveness(scores, labels, types)
    assert {"threshold", "acer_at_recommended", "eer_threshold", "eer", "auc", "basis"} <= set(rec)
    assert 0.2 < rec["threshold"] < 0.8
    assert rec["acer_at_recommended"] < 0.1
    assert rec["auc"] > 0.95


def test_recommend_threshold_fields():
    rng = np.random.default_rng(2)
    scores = np.concatenate([rng.normal(0.7, 0.1, 50), rng.normal(0.3, 0.1, 150)])
    labels = np.concatenate([np.ones(50, int), np.zeros(150, int)])
    types = np.array(["live"] * 50 + ["print"] * 50 + ["replay"] * 50 + ["cutout"] * 50)
    rec = recommend_threshold_liveness(scores, labels, types)
    assert isinstance(rec["threshold"], float)
    assert 0.0 <= rec["eer"] <= 1.0


# ---------------- score_distribution ----------------------------------------

def test_score_distribution_counts():
    scores = np.array([0.9, 0.1, 0.95, 0.05])
    labels = np.array([1, 0, 1, 0])
    d = score_distribution_liveness(scores, labels, n_bins=10)
    assert d["live"].sum() == 2
    assert d["attack"].sum() == 2
    assert len(d["bin_low"]) == 10