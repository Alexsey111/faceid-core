# scripts/calibrate_face_threshold.py — калибровка FACE_MATCH_THRESHOLD под
# целевой FAR на LFW single-face subset (релевантный проде-срез: select_main_face
# берёт ровно одно лицо, multi-involved пары в проде отклоняются — см. память
# lfw-single-face-meets-target).
#
# LFW-скрипты (run_lfw/run_single_face_eval) выбрасывают порог из tar_at_far
# (tar_001, _ = ...). Этот скрипт извлекает порог под FAR=0.001 (0.1%) и FAR=0.01
# (1%) и печатает FAR/FRR/TAR на наборе порогов-кандидатов (включая текущие
# config 0.45 и .env 0.73), чтобы выбрать operating point под цель ТЗ.
#
# Контракт 152-ФЗ: только числовые метрики, без биометрии. Кеш эмбеддингов
# строится run_lfw.py (L2-норм. ArcFace, np.dot = cosine).
#
# Использование:
#   python scripts/calibrate_face_threshold.py
#   python scripts/calibrate_face_threshold.py --target-far 0.001 --det-size 320
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from evaluation.lfw.run_lfw import cache_path, parse_pairs, _model_slug  # noqa: E402
from evaluation.lfw.run_single_face_eval import compute_face_counts  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    confusion,
    eer_point,
    recommend_thresholds,
    roc_curve,
    tar_at_far,
)


def _load_single_single_scores(det_size: int, cache_dir: Path, pairs_path: Path,
                               images_root: Path, layout: str) -> tuple[np.ndarray, np.ndarray]:
    """Повторяет логику run_single_face_eval: кеш эмбеддингов + counts →
    single-single scores/labels (оба фото содержат ровно 1 лицо)."""
    model_path = Path(settings.MODELS_DIR) / settings.ARCFACE_MODEL_REL
    slug = _model_slug(model_path)
    epath = cache_path(cache_dir, slug, det_size)
    if not epath.is_file():
        raise SystemExit(
            f"embeddings cache не найден: {epath}\n"
            f"Сначала построй кеш: python -m evaluation.lfw.run_lfw --det-size {det_size}"
        )
    with np.load(epath, allow_pickle=True) as d:
        emb_paths = list(d["paths"])
        emb_arr = np.asarray(d["embeddings"], dtype=np.float32)
    emb_map = {str(emb_paths[i]): emb_arr[i] for i in range(len(emb_paths))}

    counts = compute_face_counts(list(emb_map.keys()), det_size, cache_dir, slug, False)

    pairs = parse_pairs(pairs_path, images_root, layout=layout)
    scores, labels = [], []
    for p in pairs:
        ea, eb = emb_map.get(str(p["a"])), emb_map.get(str(p["b"]))
        if ea is None or eb is None:
            continue
        ca, cb = counts.get(str(p["a"]), 0), counts.get(str(p["b"]), 0)
        if ca == 0 or cb == 0:
            continue
        if ca == 1 and cb == 1:  # single-single срез
            scores.append(float(np.dot(ea, eb)))
            labels.append(p["label"])
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-far", type=float, default=0.001,
                    help="целевой FAR (default 0.001 = 0.1%%)")
    ap.add_argument("--det-size", type=int, default=320)
    ap.add_argument("--pairs", type=Path,
                    default=ROOT / "lfw_data" / "pairs.csv")
    ap.add_argument("--images-root", type=Path,
                    default=ROOT / "lfw_data" / "lfw")
    ap.add_argument("--layout", default="raw")
    ap.add_argument("--cache-dir", type=Path,
                    default=ROOT / "evaluation" / "lfw" / "cache")
    args = ap.parse_args()

    scores, labels = _load_single_single_scores(
        args.det_size, args.cache_dir, args.pairs, args.images_root, args.layout
    )
    n = len(scores)
    n_g = int((labels == 1).sum())
    n_i = int((labels == 0).sum())
    print(f"\nLFW single-single subset: n={n} (genuine={n_g}, impostor={n_i})")
    print(f"Цель: FAR <= {args.target_far:.4f} ({args.target_far*100:.2f}%), FRR <= 0.03 (3%)")
    print("=" * 78)

    far, tar, thr = roc_curve(scores, labels)

    # Operating point под целевой FAR (наибольший порог с far <= target).
    tar_t, thr_t = tar_at_far(far, tar, thr, target_far=args.target_far)
    # Также под FAR=0.01 (1%) для справки.
    tar_01, thr_01 = tar_at_far(far, tar, thr, target_far=0.01)
    eer, eer_thr = eer_point(far, tar, thr)
    rec = recommend_thresholds(far, tar, thr, scores, labels, target_far=args.target_far)

    print(f"\nROC operating points (single-single):")
    print(f"  AUC = {float(np.trapz(tar[np.argsort(far)], far[np.argsort(far)])):.4f}")
    print(f"  EER = {eer:.4f}  @ threshold = {eer_thr:.4f}")
    print(f"  TAR@FAR={args.target_far:.4f} = {tar_t:.4f}  @ threshold = {thr_t:.4f}  <-- operating point")
    print(f"  TAR@FAR=0.0100       = {tar_01:.4f}  @ threshold = {thr_01:.4f}")

    # FAR/FRR на наборе порогов-кандидатов.
    candidates = sorted({
        round(thr_t, 4),
        round(thr_01, 4),
        round(eer_thr, 4),
        round(float(rec["high"]), 4),
        0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.73, 0.80,
    })
    print(f"\n{'threshold':>10} {'FAR':>8} {'FRR':>8} {'TAR':>8} {'TP':>5} {'FP':>5} {'FN':>5}  verdict")
    print("-" * 78)
    target_far = args.target_far
    for t in candidates:
        m = confusion(scores, labels, t)
        far_ok = m["far"] <= target_far
        frr_ok = m["frr"] <= 0.03
        verdict = "OK" if (far_ok and frr_ok) else (
            "FAR>цель" if not far_ok else "FRR>3%"
        )
        # Подсветка текущих порогов config/.env
        tag = ""
        if abs(t - 0.45) < 1e-9:
            tag = "  <- config.py default (FRR<=3% калибровка)"
        elif abs(t - 0.73) < 1e-9:
            tag = "  <- .env значение (расхождение)"
        print(f"{t:>10.4f} {m['far']:>8.4f} {m['frr']:>8.4f} {m['tar']:>8.4f} "
              f"{m['tp']:>5d} {m['fp']:>5d} {m['fn']:>5d}  {verdict}{tag}")

    print("-" * 78)
    print("\nРекомендация:")
    print(f"  Operating point под FAR<={target_far:.4f}: threshold = {thr_t:.4f} "
          f"(FAR={confusion(scores, labels, thr_t)['far']:.4f}, "
          f"FRR={confusion(scores, labels, thr_t)['frr']:.4f})")
    print(f"  Текущий config FACE_MATCH_THRESHOLD = {settings.FACE_MATCH_THRESHOLD}")
    cur = confusion(scores, labels, settings.FACE_MATCH_THRESHOLD)
    print(f"    при {settings.FACE_MATCH_THRESHOLD}: FAR={cur['far']:.4f} FRR={cur['frr']:.4f} "
          f"{'(удовлетворяет FAR<=цель И FRR<=3%)' if (cur['far'] <= target_far and cur['frr'] <= 0.03) else '(НЕ удовлетворяет)'}")
    print("\nВНИМАНИЕ: LFW-pairs != СКУД-сценарий (эталон-регистрация + кадр-с-камеры-прохода).")
    print("Калибровка на LFW — верхняя оценка порога; реальный FRR в проде выше.")
    print("Реал-валидация на целевых парах — future (нужны данные).")


if __name__ == "__main__":
    main()