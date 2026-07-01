# tests/evaluation/liveness/test_liveness_protocols.py — unit-тесты eval_liveness (pure numpy).

import numpy as np
import pytest

from evaluation.liveness.protocols import eval_liveness

pytestmark = pytest.mark.unit


def _make_dataset():
    """live ~0.8, attack ~0.2; 3 типа атак поровну. Идеальное разделение при thr≈0.5."""
    rng = np.random.default_rng(7)
    n_live = 60
    n_attack = 90  # 30 каждого типа
    scores = np.concatenate([
        rng.normal(0.8, 0.05, n_live),
        rng.normal(0.2, 0.05, n_attack),
    ])
    labels = np.concatenate([np.ones(n_live, int), np.zeros(n_attack, int)])
    types = np.array(
        ["live"] * n_live
        + ["print"] * 30
        + ["replay"] * 30
        + ["cutout"] * 30
    )
    return scores, labels, types


def test_eval_liveness_structure():
    scores, labels, types = _make_dataset()
    r = eval_liveness(scores, labels, types)
    for k in ("n_live", "n_attack", "current_threshold", "recommended",
             "at_current", "at_recommended", "roc", "score_dist"):
        assert k in r
    assert r["n_live"] == 60 and r["n_attack"] == 90


def test_eval_liveness_at_current_threshold_default_05():
    scores, labels, types = _make_dataset()
    r = eval_liveness(scores, labels, types)
    assert r["at_current"]["threshold"] == pytest.approx(0.5)
    for k in ("apcer_max", "npcer", "acer", "accuracy", "apcer_per_type"):
        assert k in r["at_current"]


def test_eval_liveness_at_current_custom_threshold():
    scores, labels, types = _make_dataset()
    r = eval_liveness(scores, labels, types, current_threshold=0.3)
    assert r["at_current"]["threshold"] == pytest.approx(0.3)
    assert r["current_threshold"] == pytest.approx(0.3)


def test_eval_liveness_per_type_table():
    scores, labels, types = _make_dataset()
    r = eval_liveness(scores, labels, types)
    # per-type в at_current содержит все 3 типа с n и apcer
    ap = r["at_current"]["apcer_per_type"]
    assert set(ap.keys()) == {"print", "replay", "cutout"}
    for atype in ("print", "replay", "cutout"):
        assert ap[atype]["n"] == 30
        # идеальное разделение при 0.5 → APCER каждого типа ≈ 0
        assert ap[atype]["apcer"] < 0.1


def test_eval_liveness_recommended_minimizes_acer():
    scores, labels, types = _make_dataset()
    r = eval_liveness(scores, labels, types)
    rec = r["recommended"]
    assert {"threshold", "acer_at_recommended", "eer", "auc"} <= set(rec)
    assert 0.2 < rec["threshold"] < 0.8
    assert rec["acer_at_recommended"] < 0.1
    # recommended не хуже current 0.5 (оба идеальны здесь) ИЛИ просто низкий
    assert r["at_recommended"]["acer"] <= r["at_current"]["acer"] + 1e-6 or \
           r["at_recommended"]["acer"] < 0.1


def test_eval_liveness_acer_overall_formula():
    scores, labels, types = _make_dataset()
    r = eval_liveness(scores, labels, types)
    cur = r["at_current"]
    # ACER overall = (apcer_max + npcer) / 2
    assert cur["acer"] == pytest.approx((cur["apcer_max"] + cur["npcer"]) / 2.0)


def test_eval_liveness_missing_class_returns_error():
    # только live, нет атак → graceful error
    scores = np.array([0.9, 0.8])
    labels = np.array([1, 1])
    types = np.array(["live", "live"])
    r = eval_liveness(scores, labels, types)
    assert "error" in r
    assert r["n_attack"] == 0


def test_eval_liveness_high_threshold_rejects_live():
    # порог 1.5 → все live отбракованы (NPCER=1), атаки не пропущены (APCER=0)
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    types = np.array(["live", "live", "print", "print"])
    r = eval_liveness(scores, labels, types, current_threshold=1.5)
    assert r["at_current"]["npcer"] == pytest.approx(1.0)
    assert r["at_current"]["apcer_max"] == pytest.approx(0.0)