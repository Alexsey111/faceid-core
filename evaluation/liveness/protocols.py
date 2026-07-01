# evaluation/liveness/protocols.py — протокол оценки liveness (сводные + per-type).

from __future__ import annotations

import numpy as np

from evaluation.liveness.metrics import (
    CURRENT_LIVENESS_THRESHOLD,
    acer_overall,
    apcer_per_type,
    confusion_liveness,
    recommend_threshold_liveness,
    roc_liveness,
    score_distribution_liveness,
)


def _eval_at(scores, labels, attack_types, threshold):
    """Сводка метрик при фиксированном пороге: per-type APCER, max APCER, NPCER, ACER, accuracy."""
    conf = confusion_liveness(scores, labels, threshold)
    per_type = apcer_per_type(scores, labels, attack_types, threshold)
    apcers = [pt["apcer"] for pt in per_type.values() if pt["n"] > 0]
    apcer_max = max(apcers) if apcers else 0.0
    npcer = conf["npcer"]
    return {
        "threshold": float(threshold),
        "apcer_per_type": {k: {"n": v["n"], "apcer": v["apcer"]} for k, v in per_type.items()},
        "apcer_max": apcer_max,
        "npcer": npcer,
        "acer": acer_overall(apcer_max, npcer),
        "accuracy": conf["accuracy"],
        "n_live": int(np.sum(labels == 1)),
        "n_attack": int(np.sum(labels == 0)),
    }


def eval_liveness(
    scores: np.ndarray,
    labels: np.ndarray,
    attack_types: np.ndarray,
    current_threshold: float = CURRENT_LIVENESS_THRESHOLD,
) -> dict:
    """
    Сводная оценка liveness. REPORT-ONLY.
    Возвращает: n_live/n_attack, recommended (thr/acer/eer/auc), at_current, at_recommended,
    per_type (на recommended), roc, score_dist.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    attack_types = np.asarray(attack_types)

    n_live = int(np.sum(labels == 1))
    n_attack = int(np.sum(labels == 0))
    if n_live == 0 or n_attack == 0:
        return {"error": "need both live and attack samples", "n_live": n_live, "n_attack": n_attack}

    rec = recommend_threshold_liveness(scores, labels, attack_types)
    at_current = _eval_at(scores, labels, attack_types, current_threshold)
    at_recommended = _eval_at(scores, labels, attack_types, rec["threshold"])

    apcer, tpr, thr_roc = roc_liveness(scores, labels)

    return {
        "n_live": n_live,
        "n_attack": n_attack,
        "current_threshold": current_threshold,
        "recommended": rec,
        "at_current": at_current,
        "at_recommended": at_recommended,
        "roc": {"apcer": apcer, "tpr": tpr, "thresholds": thr_roc},
        "score_dist": score_distribution_liveness(scores, labels),
        "scores": scores,
        "labels": labels,
        "attack_types": attack_types,
    }