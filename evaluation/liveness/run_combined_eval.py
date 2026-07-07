# evaluation/liveness/run_combined_eval.py — CLI: combined passive+active liveness eval.
#
# Запуск из кеша run_eval.py (без моделей/детектора): грузит .npz frame-scores →
# eval_combined (frame-passive + video-passive-temporal + active-gate-policy) →
# детерминированный JSON-отчёт.
#
# Использование:
#   python -m evaluation.liveness.run_combined_eval \
#       --cache evaluation/liveness/cache/liveness_full_nfr30_det320_yakhyo_v2.npz \
#       --threshold 0.859
# (threshold = settings.LIVENESS_THRESHOLD; без --threshold — baseline 0.5)
#
# Повторный прогон из того же кеша → байт-идентичный JSON (numpy-детерминизм, sorted keys).

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# ensure_models_env вызывать НЕ нужно: combined.py не импортирует app.* (pure numpy).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # git-root on path

from evaluation.liveness.combined import eval_combined  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("liveness.run_combined_eval")

OUT_DIR_DEFAULT = "evaluation/liveness/out"


def _jsonable(obj):
    """Рекурсивное приведение numpy-типов к JSON-сериализуемым (без сырых массивов)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Combined passive+active liveness eval (из кеша)")
    parser.add_argument("--cache", required=True, help="путь к .npz кешу run_eval.py")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="порог passive (settings.LIVENESS_THRESHOLD; default 0.5 baseline)")
    parser.add_argument("--out", default=OUT_DIR_DEFAULT)
    parser.add_argument("--report-name", default="liveness_combined")
    args = parser.parse_args(argv)

    cache_path = Path(args.cache)
    if not cache_path.is_file():
        raise SystemExit(f"cache not found: {cache_path}")

    # Грузим напрямую np.load: load_predictions() собирает путь из (name,n_frames,
    # det_size,slug), но нам передают уже готовый .npz — проще прочитать как есть.
    with np.load(cache_path, allow_pickle=True) as d:
        meta = json.loads(str(d["meta_json"].item()))
        scores = np.array(d["scores"], dtype=np.float64)
        labels = np.array(d["labels"], dtype=np.int64)
        attack_types = np.array(d["attack_types"], dtype=object)
        sources = np.array(d["sources"], dtype=object)

    logger.info("cache: %s (%d scored frames, model=%s)",
                cache_path, len(scores), meta.get("model_slug", "?"))

    result = eval_combined(scores, labels, attack_types, sources, current_threshold=args.threshold)

    # dataset_meta (без volatile-полей, без абсолютных путей — только счётчики).
    dataset_meta = {
        "n_scored_frames": int(len(scores)),
        "n_live_frames": int(np.sum(labels == 1)),
        "n_attack_frames": int(np.sum(labels == 0)),
        "n_videos_total": int(result["video"]["n_videos"]),
        "passive_threshold": float(args.threshold),
        "model_slug": meta.get("model_slug", ""),
        "model_desc": meta.get("model_desc", ""),
        "cache_file": cache_path.name,
    }

    payload = {
        "dataset_meta": _jsonable(dataset_meta),
        "frame_passive": _jsonable(result["frame"]),
        "video_passive_temporal": _jsonable(result["video"]),
        "active_gate_policy": _jsonable(result["active_gate"]),
        "kpi": _jsonable(result["kpi"]),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.report_name}_report.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
                         encoding="utf-8")
    logger.info("report: %s", json_path)

    _print_summary(result, dataset_meta)


def _print_summary(result: dict, meta: dict) -> None:
    f = result["frame"]
    v = result["video"]
    ag = result["active_gate"]
    k = result["kpi"]
    print("=" * 74)
    print("Combined liveness eval - passive frame + passive video-temporal + active-gate")
    print("-" * 74)
    print(f"  frames: {meta['n_scored_frames']} (live {meta['n_live_frames']}, "
          f"attack {meta['n_attack_frames']}); videos: {v['n_videos']}; "
          f"thr={meta['passive_threshold']}")
    print("-" * 74)
    print("  [1] FRAME passive (baseline):")
    print(f"      accuracy={f['accuracy']:.4f}  APCER={f['apcer']:.4f}  NPCER={f['npcer']:.4f}  "
          f"APCER_max={f['apcer_max']:.4f}  AUC={f['auc']:.4f}")
    for at, pt in f["apcer_per_type"].items():
        print(f"        {at:8s} n={pt['n']:4d}  APCER={pt['apcer']:.4f}")
    print("  [2] VIDEO passive temporal (mean real_score over 30 frames):")
    print(f"      accuracy={v['accuracy']:.4f}  APCER={v['apcer']:.4f}  NPCER={v['npcer']:.4f}  "
          f"APCER_max={v['apcer_max']:.4f}  AUC={v['auc']:.4f}")
    for at, pt in v["apcer_per_type"].items():
        print(f"        {at:8s} n={pt['n']:4d}  APCER={pt['apcer']:.4f}")
    print(f"      mean std real_score: live={v['mean_std_real_score']['live']:.4f} "
          f"attack={v['mean_std_real_score']['attack']:.4f} (proxy micro-motion)")
    print("  [3] ACTIVE-GATE policy (LIVENESS_ACTIVE_REQUIRED=true):")
    print(f"      spoof_accept_rate={ag['spoof_accept_rate']:.4f}  "
          f"spoof_rejection_rate={ag['spoof_rejection_rate']:.4f}  "
          f"cutout APCER: {ag['cutout_apcer_passive_baseline']:.4f}->{ag['cutout_apcer_active_gate']:.4f}")
    print("-" * 74)
    print("  KPI by TZ 'Liveness >=98%' (security goal = spoof-rejection):")
    print(f"      frame passive acc @ current      = {k['frame_passive_accuracy_at_current']:.4f}  "
          f"[{'MEET' if k['frame_passive_meets_98'] else 'MISS'}]")
    print(f"      video temporal acc @ current     = {k['video_passive_temporal_accuracy_at_current']:.4f}  "
          f"[{'MEET' if k['video_passive_temporal_meets_98_at_current'] else 'MISS'}]")
    print(f"      video temporal acc @ recommended = {k['video_passive_temporal_accuracy_at_recommended']:.4f}  "
          f"(thr={k['video_passive_temporal_recommended_threshold']:.4f}, AUC={k['video_passive_temporal_auc']:.4f})  "
          f"[{'MEET' if k['video_passive_temporal_meets_98_at_recommended'] else 'MISS'}]")
    print(f"      active-gate spoof rejection      = {k['active_gate_spoof_rejection']:.4f}  "
          f"[{'MEET' if k['active_gate_spoof_rejection_meets_98'] else 'MISS'}]")
    sep = k['video_passive_temporal_separation']
    print(f"      separation: live_min={sep['live_score_min']:.4f} attack_max={sep['attack_score_max']:.4f} "
          f"perfect_gap={sep['perfect_gap']}  (n_video={k['n_video_level_samples']}, small-sample!)")
    print("=" * 74)


if __name__ == "__main__":
    main()