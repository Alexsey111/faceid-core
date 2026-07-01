# evaluation/liveness/metrics.py — метрики anti-spoofing (ISO/IEC 30107-3).
#
# Чистый numpy, БЕЗ импорта app. Переиспользует roc_curve/eer/auc из evaluation.metrics
# (тот же пакет) с переинтерпретацией осей: far → APCER, tar → 1-NPCER.
#
# Контракт: scores — real_score (чем выше, тем «живее»); labels — 1=live, 0=attack;
# attack_types — строковый тип атаки для attack-семплов ('print'|'replay'|'cutout'),
# для live-семплов значение игнорируется (обычно 'live').
# Решение live: score >= threshold.

from __future__ import annotations

from typing import Any

import numpy as np

from evaluation.metrics import auc_from_roc, eer_point, roc_curve

CURRENT_LIVENESS_THRESHOLD = 0.5
ATTACK_TYPES = ("print", "replay", "cutout")


def confusion_liveness(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    """
    TP/FP/TN/FN + APCER/NPCER/accuracy при фиксированном пороге.
      APCER = FP/(FP+TN)  — доля атак, пропущенных как live.
      NPCER = FN/(FN+TP)  — доля live, отбракованных как spoof.
      accuracy = (TP+TN)/total.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    predicted_live = scores >= threshold
    actual_live = labels == 1

    tp = int(np.sum(predicted_live & actual_live))     # live accepted
    fp = int(np.sum(predicted_live & ~actual_live))    # attack accepted (false)
    tn = int(np.sum(~predicted_live & ~actual_live))   # attack rejected
    fn = int(np.sum(~predicted_live & actual_live))   # live rejected

    apcer = fp / (fp + tn) if (fp + tn) else 0.0
    npcer = fn / (fn + tp) if (fn + tp) else 0.0
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "apcer": apcer, "npcer": npcer,
        "acer": (apcer + npcer) / 2.0,
        "accuracy": accuracy,
        "tpr": tp / (tp + fn) if (tp + fn) else 0.0,  # 1-NPCER (live accepted)
    }


def apcer_per_type(
    scores: np.ndarray, labels: np.ndarray, attack_types: np.ndarray, threshold: float
) -> dict[str, dict[str, float]]:
    """
    APCER для каждого типа атаки отдельно: APCER_type = P(score>=thr | attack этого типа).
    attack-семплы: только label==0. live-семплы в расчёт per-type не входят.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    attack_types = np.asarray(attack_types)
    attack_mask = labels == 0
    out: dict[str, dict[str, float]] = {}
    for atype in ATTACK_TYPES:
        sel = attack_mask & (attack_types == atype)
        n = int(np.sum(sel))
        if n == 0:
            out[atype] = {"n": 0, "apcer": 0.0, "n_accepted": 0}
            continue
        accepted = int(np.sum(scores[sel] >= threshold))
        out[atype] = {
            "n": n,
            "accepted": accepted,
            "apcer": accepted / n,
        }
    return out


def acer_overall(apcer_max: float, npcer: float) -> float:
    """ACER (ISO) = (max APCER по типам атак + NPCER) / 2."""
    return (apcer_max + npcer) / 2.0


def roc_liveness(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ROC: (apcer, 1-npcer, thresholds) — apcer по возрастанию.
    1-npcer = TPR = доля live, принятых как live.
    """
    far, tar, thr = roc_curve(scores, labels)  # labels: 1=live
    # far == APCER (attack accepted rate), tar == TPR == 1-NPCER.
    return far, tar, thr


def recommend_threshold_liveness(
    scores: np.ndarray, labels: np.ndarray, attack_types: np.ndarray
) -> dict[str, Any]:
    """
    Порог, минимизирующий ACER overall = (max_apcer_over_types + npcer)/2.
    По сетке 200 порогов. Дополнительно thr@EER. REPORT-ONLY.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    attack_types = np.asarray(attack_types)

    grid = np.linspace(float(np.min(scores)), float(np.max(scores)), 200)
    best_thr = float(grid[0])
    best_acer = float("inf")
    best_apcer_max = 0.0
    best_npcer = 0.0
    for t in grid:
        per_type = apcer_per_type(scores, labels, attack_types, t)
        apcers = [pt["apcer"] for pt in per_type.values() if pt["n"] > 0]
        apcer_max = max(apcers) if apcers else 0.0
        npcer = confusion_liveness(scores, labels, t)["npcer"]
        acer = acer_overall(apcer_max, npcer)
        if acer < best_acer:
            best_acer = acer
            best_thr = float(t)
            best_apcer_max = apcer_max
            best_npcer = npcer

    far, tar, thr_roc = roc_liveness(scores, labels)
    eer, eer_thr = eer_point(far, tar, thr_roc)
    auc = auc_from_roc(far, tar)

    return {
        "threshold": best_thr,
        "acer_at_recommended": best_acer,
        "apcer_max_at_recommended": best_apcer_max,
        "npcer_at_recommended": best_npcer,
        "eer_threshold": float(eer_thr),
        "eer": eer,
        "auc": auc,
        "basis": (
            "recommended = argmin ACER_overall = (max_apcer_over_types + npcer)/2 "
            "over 200-point grid; eer=|apcer-(1-tpr)| min; auc=trapz"
        ),
    }


def score_distribution_liveness(
    scores: np.ndarray, labels: np.ndarray, n_bins: int = 50
) -> dict[str, np.ndarray]:
    """Гистограммы real_score для live (label=1) и attack (label=0). Диапазон [0,1] (softmax)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    live, _ = np.histogram(scores[labels == 1], bins=bins)
    attack, _ = np.histogram(scores[labels == 0], bins=bins)
    return {"bin_low": bins[:-1], "bin_high": bins[1:], "live": live, "attack": attack}