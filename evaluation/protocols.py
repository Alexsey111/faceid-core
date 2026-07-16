# evaluation/protocols.py — протоколы оценки точности 1:1 и 1:N + faiss-консистентность.
#
# 1:1 (verification): genuine/impostor пары → FAR/FRR/TAR@FAR/EER/AUC/ROC +
#   рекомендация порогов (REPORT-ONLY) и сравнение с текущими оперативными порогами.
# 1:N (identification): gallery/probe split → CMC, rank-1/rank-5.
# validate_faiss_consistency: сверка ранжирования numpy vs FaissIndex (IndexFlatIP).

from __future__ import annotations

import numpy as np

from evaluation.metrics import (
    auc_from_roc,
    cmc_curve,
    confusion,
    eer_point,
    recommend_thresholds,
    roc_curve,
    score_distribution,
    tar_at_far,
)
from evaluation.pairs import build_pairs_1to1, gallery_probe_split, pair_scores

# Оперативные пороги пайплайна (app/core.config) — для сравнения в отчёте.
# Держать синхронно с app/core/config.py: FACE_MATCH_THRESHOLD / FACE_LOW_THRESHOLD
# / FACE_MARGIN_THRESHOLD. evaluation/ — чистый harness без импорта app, поэтому
# хардкод (не pydantic-settings); при смене порога в config — обновить здесь.
CURRENT_HIGH_THRESHOLD = 0.45
CURRENT_LOW_THRESHOLD = 0.30
CURRENT_MARGIN_THRESHOLD = 0.10
TARGET_FAR = 0.001


def eval_1to1(
    embeddings: np.ndarray,
    ids: list[str] | np.ndarray,
    impostor_ratio: int = 10,
    seed: int = 42,
    target_far: float = TARGET_FAR,
    current_high: float = CURRENT_HIGH_THRESHOLD,
) -> dict:
    """
    1:1 verification. Возвращает сводные метрики + рекомендованные/текущие пороги.
    Эмбеддинги L2-нормализованы → cosine = dot. REPORT-ONLY.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    ids_arr = np.asarray(ids)

    idx_i, idx_j, labels = build_pairs_1to1(ids_arr, impostor_ratio=impostor_ratio, seed=seed)
    if len(labels) == 0:
        return {"error": "no pairs (need >=2 samples for some id)"}

    scores = pair_scores(embeddings, idx_i, idx_j)
    far, tar, thr = roc_curve(scores, labels)
    auc = auc_from_roc(far, tar)
    tar_at_target, high_thr = tar_at_far(far, tar, thr, target_far)
    eer, eer_thr = eer_point(far, tar, thr)
    rec = recommend_thresholds(far, tar, thr, scores, labels, target_far=target_far)

    n_genuine = int((labels == 1).sum())
    n_impostor = int((labels == 0).sum())

    # На рекомендованном high-пороге и на текущем оперативном (0.60).
    at_recommended_high = confusion(scores, labels, rec["high"])
    at_current = confusion(scores, labels, current_high)

    return {
        "n_genuine": n_genuine,
        "n_impostor": n_impostor,
        "impostor_ratio": impostor_ratio,
        "seed": seed,
        "target_far": target_far,
        "tar_at_far": tar_at_target,
        "frr_at_recommended_high": at_recommended_high["frr"],
        "far_at_recommended_high": at_recommended_high["far"],
        "eer": eer,
        "auc": auc,
        "thresholds": {
            "recommended": rec,
            "current": {
                "high": current_high,
                "low": CURRENT_LOW_THRESHOLD,
                "margin": CURRENT_MARGIN_THRESHOLD,
            },
        },
        "at_current_high": {
            "far": at_current["far"],
            "frr": at_current["frr"],
            "tar": at_current["tar"],
            "threshold": current_high,
        },
        "roc": {"far": far, "tar": tar, "frr": 1.0 - tar, "thresholds": thr},
        "score_dist": score_distribution(scores, labels),
        # сырые scores/labels — для воспроизводимости и доп. CSV.
        "scores": scores,
        "labels": labels,
    }


def eval_1toN(
    embeddings: np.ndarray,
    ids: list[str] | np.ndarray,
    max_rank: int = 50,
) -> dict:
    """
    1:N identification. Gallery = первый сэмпл каждого id, probe = остальные.
    Ids с одним фото → gallery-only (не queried). Возвращает CMC + rank-1/rank-5.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    ids_arr = np.asarray(ids)

    gallery_idx, probe_idx = gallery_probe_split(ids_arr)
    if len(probe_idx) == 0 or len(gallery_idx) == 0:
        return {"error": "empty gallery or probe"}

    g_emb = embeddings[gallery_idx]
    g_ids = ids_arr[gallery_idx]
    p_emb = embeddings[probe_idx]
    p_ids = ids_arr[probe_idx]

    n_total_ids = len(np.unique(ids_arr))
    n_gallery = len(gallery_idx)
    # id с ровно 1 фото → попадают только в gallery (не queried в 1:N).
    _unique, _counts = np.unique(ids_arr, return_counts=True)
    n_ids_single_image = int(np.sum(_counts == 1))

    ranks, acc = cmc_curve(p_emb, p_ids, g_emb, g_ids, max_rank=max_rank)
    rank1 = float(acc[0])
    rank5 = float(acc[4]) if len(acc) >= 5 else float(acc[-1])

    return {
        "n_gallery": int(n_gallery),
        "n_probes": int(len(probe_idx)),
        "n_ids_total": int(n_total_ids),
        "n_ids_single_image": int(n_ids_single_image),
        "max_rank": int(max_rank),
        "rank1": rank1,
        "rank5": rank5,
        "cmc": {"ranks": ranks, "accuracy": acc},
    }


def validate_faiss_consistency(
    embeddings: np.ndarray,
    ids: list[str] | np.ndarray,
    max_rank: int = 5,
) -> dict:
    """
    Сверка ранжирования numpy (probe @ gallery.T) vs FaissIndex (IndexFlatIP, cosine).
    Возвращает {n_probes, mismatch, max_rank}, mismatch = число probe, у которых
    top-1 gallery-индекс от faiss отличается от numpy. Импортирует faiss лениво.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    ids_arr = np.asarray(ids)
    gallery_idx, probe_idx = gallery_probe_split(ids_arr)
    if len(probe_idx) == 0 or len(gallery_idx) == 0:
        return {"error": "empty gallery or probe"}

    from app.services.faiss_index import FaissIndex  # ленивый импорт (опц. зависимость)

    g_emb = embeddings[gallery_idx].astype(np.float32)
    p_emb = embeddings[probe_idx].astype(np.float32)
    g_ids_int = np.arange(len(gallery_idx), dtype=np.int64)  # gallery idx → int

    index = FaissIndex(dim=int(g_emb.shape[1]))
    index.add(g_emb, [int(x) for x in g_ids_int])

    # numpy-ранжирование (эталон): probe @ gallery.T → argsort desc.
    np_scores = p_emb @ g_emb.T
    np_top1 = np.argmax(np_scores, axis=1)

    mismatch = 0
    for i in range(len(p_emb)):
        res = index.search(p_emb[i], k=max_rank)
        if not res:
            mismatch += 1
            continue
        faiss_top1 = int(res[0]["user_id"])
        if faiss_top1 != int(np_top1[i]):
            mismatch += 1

    return {
        "n_probes": int(len(p_emb)),
        "n_gallery": int(len(g_emb)),
        "max_rank": int(max_rank),
        "mismatch": int(mismatch),
    }