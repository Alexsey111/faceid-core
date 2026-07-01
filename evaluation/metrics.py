# evaluation/metrics.py — метрики точности recognition (чистый numpy, БЕЗ импорта app).
#
# Назначение: детерминированный расчёт FAR/FRR/TAR/EER/AUC/ROC/CMC для eval-harness.
# Не зависит от app/core/config и БД — это позволяет гонять unit-тесты без инфры.
#
# Контракт: scores — float-массив косинус-сходств; labels — int (1=genuine, 0=impostor).
# Эмбеддинги предполагаются L2-нормализованными (контракт OnnxArcFaceEncoder),
# поэтому косинус = скалярное произведение; диапазон scores ~ [-1, 1].

from __future__ import annotations

from typing import Any

import numpy as np

# Порог FAR для основной метрики ТЗ: TAR @ FAR ≤ 0.1%.
TARGET_FAR = 0.001


def confusion(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    """
    TP/FP/TN/FN + FAR/FRRT/TAR при фиксированном пороге.
    Решение genuine: score >= threshold.
      FAR = FP/(FP+TN)  — доля impostor, пропущенных как genuine.
      FRR = FN/(FN+TP)  — доля genuine, отбракованных как impostor.
      TAR = 1 - FRR     — доля genuine, пропущенных верно.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    predicted_positive = scores >= threshold
    actual_positive = labels == 1

    tp = int(np.sum(predicted_positive & actual_positive))
    fp = int(np.sum(predicted_positive & ~actual_positive))
    tn = int(np.sum(~predicted_positive & ~actual_positive))
    fn = int(np.sum(~predicted_positive & actual_positive))

    far = fp / (fp + tn) if (fp + tn) else 0.0
    frr = fn / (fn + tp) if (fn + tp) else 0.0
    tar = 1.0 - frr

    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "far": far, "frr": frr, "tar": tar,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }


def roc_curve(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ROC: возвращает (far, tar, thresholds), far по возрастанию.
    Строится скольжением порога по всем уникальным score-точкам (descending).
    Граничные точки: (far=0,tar=0) при threshold=+inf и (far=1,tar=1) при хвосте.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    n_positive = int(np.sum(labels == 1))
    n_negative = int(np.sum(labels == 0))
    if n_positive == 0 or n_negative == 0:
        # Вырожденный случай — ROC не определён. Возвращаем тривиальную кривую.
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])

    # Сортируем по убыванию score (stable mergesort — детерминизм при равных score).
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    l_sorted = labels[order]

    tp_cum = np.cumsum(l_sorted == 1)
    fp_cum = np.cumsum(l_sorted == 0)
    tar = tp_cum / n_positive
    far = fp_cum / n_negative

    # Границы: начало (порог выше любого score — ничего не пропускаем) и конец.
    far = np.concatenate([[0.0], far, [1.0]])
    tar = np.concatenate([[0.0], tar, [1.0]])
    # Пороги: +inf в начале (ничего не positive), последний score чуть ниже в хвосте.
    thr = np.concatenate(
        [[s_sorted[0] + 1.0], s_sorted, [s_sorted[-1] - 1.0]]
    )
    return far, tar, thr


def auc_from_roc(far: np.ndarray, tar: np.ndarray) -> float:
    """AUC через трапеции по far-возрастающей кривой."""
    far = np.asarray(far, dtype=np.float64)
    tar = np.asarray(tar, dtype=np.float64)
    order = np.argsort(far, kind="mergesort")
    return float(np.trapz(tar[order], far[order]))


def tar_at_far(
    far: np.ndarray, tar: np.ndarray, thresholds: np.ndarray, target_far: float = TARGET_FAR
) -> tuple[float, float]:
    """
    (tar, threshold) при наибольшем пороге с far ≤ target_far (tie-break: max tar,
    затем max threshold). Возвращает (0.0, thresholds[0]) если ни один порог не даёт
    far ≤ target.
    """
    far = np.asarray(far, dtype=np.float64)
    tar = np.asarray(tar, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)

    mask = far <= target_far
    if not mask.any():
        return 0.0, float(thresholds[0])
    idx = np.where(mask)[0]
    # Среди допустимых — максимальный tar; при равенстве — максимальный threshold.
    best = idx[np.lexsort((-thresholds[idx], -tar[idx]))[0]]
    return float(tar[best]), float(thresholds[best])


def eer_point(
    far: np.ndarray, tar: np.ndarray, thresholds: np.ndarray
) -> tuple[float, float]:
    """
    (eer, threshold) — точка, где FAR == FRR (= 1 - TAR). EER = среднее |far - frr|.
    """
    far = np.asarray(far, dtype=np.float64)
    tar = np.asarray(tar, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    frr = 1.0 - tar
    diff = np.abs(far - frr)
    i = int(np.argmin(diff))
    eer = float((far[i] + frr[i]) / 2.0)
    return eer, float(thresholds[i])


def f1_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    m = confusion(scores, labels, threshold)
    p, r = m["precision"], m["recall"]
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def recommend_thresholds(
    far: np.ndarray,
    tar: np.ndarray,
    thresholds: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    target_far: float = TARGET_FAR,
) -> dict[str, Any]:
    """
    Рекомендация порогов (REPORT-ONLY, не пишется в config):
      high  = threshold @ TAR@FAR=target_far (оперативный порог ТЗ);
      low   = threshold, максимизирующий F1 (граница «uncertain»-зоны);
      margin = clamp(high - low, 0, high);
      eer   = threshold в точке EER.
    """
    _, high = tar_at_far(far, tar, thresholds, target_far)

    # low: max F1 по сетке порогов (не по всем scores — быстрее, достаточно плотно).
    grid = np.linspace(float(np.min(scores)), float(np.max(scores)), 200)
    f1s = np.array([f1_at_threshold(scores, labels, t) for t in grid])
    low = float(grid[int(np.argmax(f1s))])

    margin = float(max(0.0, min(high - low, high)))
    _, eer_thr = eer_point(far, tar, thresholds)

    return {
        "high": float(high),
        "low": low,
        "margin": margin,
        "eer": float(eer_thr),
        "basis": (
            f"high=TAR@FAR={target_far}; low=max-F1 over 200-point grid; "
            f"margin=clamp(high-low,0,high); eer=|far-frr| min"
        ),
    }


def cmc_curve(
    probe_emb: np.ndarray,
    probe_ids: np.ndarray,
    gallery_emb: np.ndarray,
    gallery_ids: np.ndarray,
    max_rank: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    CMC: (ranks 1..max_rank, accuracy_at_rank).
    scores = probe_emb @ gallery_emb.T (косинус, эмбеддинги L2-норм.).
    rank probe = позиция первой gallery-записи с тем же id (1-based).
    accuracy_at_rank[r] = mean(rank <= r).
    """
    probe_emb = np.asarray(probe_emb, dtype=np.float32)
    gallery_emb = np.asarray(gallery_emb, dtype=np.float32)
    probe_ids = np.asarray(probe_ids)
    gallery_ids = np.asarray(gallery_ids)

    scores = probe_emb @ gallery_emb.T  # (P, G)
    # Сортируем gallery по убыванию score для каждого probe.
    order = np.argsort(-scores, axis=1, kind="mergesort")
    ranked_ids = gallery_ids[order]  # (P, G)

    max_rank = int(min(max_rank, len(gallery_ids)))
    ranks = np.arange(1, max_rank + 1)
    acc = np.empty(max_rank, dtype=np.float64)
    for r in ranks:
        # True, если true-id встречается среди первых r позиций.
        hit = np.any(ranked_ids[:, :r] == probe_ids[:, None], axis=1)
        acc[r - 1] = float(np.mean(hit))
    return ranks, acc


def score_distribution(
    scores: np.ndarray, labels: np.ndarray, n_bins: int = 50
) -> dict[str, np.ndarray]:
    """
    Гистограммы распределения score для genuine (label=1) и impostor (label=0).
    Возвращает {bin_low, bin_high, genuine, impostor} по общему диапазону [-1, 1].
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    bins = np.linspace(-1.0, 1.0, n_bins + 1)
    genuine, _ = np.histogram(scores[labels == 1], bins=bins)
    impostor, _ = np.histogram(scores[labels == 0], bins=bins)
    return {
        "bin_low": bins[:-1],
        "bin_high": bins[1:],
        "genuine": genuine,
        "impostor": impostor,
    }