# evaluation/run_1to1.py — CLI 1:1 verification (FAR/FRR/TAR@FAR/EER/AUC/ROC).
#
#   python -m evaluation.run_1to1 --dataset {orig|extracted}
#       [--cache-dir evaluation/cache] [--det-size 512]
#       [--impostor-ratio 10] [--seed 42] [--out evaluation/out] [--report-name NAME]
#
# Грузит кеш эмбеддингов → eval_1to1 → JSON+CSV. Пороги REPORT-ONLY (config не трогает).

from __future__ import annotations

import argparse

from evaluation.cli_common import (
    add_dataset_arg,
    cache_dir_arg,
    dataset_meta_from_cache,
    det_size_for,
    load_cache,
)
from evaluation.protocols import eval_1to1
from evaluation.report import OUT_DIR_DEFAULT, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="1:1 face verification eval")
    add_dataset_arg(parser)
    cache_dir_arg(parser)
    parser.add_argument("--det-size", type=int, default=512, help="должен совпадать с кешем orig")
    parser.add_argument("--impostor-ratio", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=OUT_DIR_DEFAULT)
    parser.add_argument("--report-name", default=None)
    args = parser.parse_args()

    det_size = det_size_for(args.dataset, args.det_size)
    name = args.report_name or f"{args.dataset}_1to1"

    cached = load_cache(args.cache_dir, args.dataset, det_size)
    meta = dataset_meta_from_cache(cached)
    print(f"[run_1to1] {name}: n_images={meta.get('n_images')} n_ids={meta.get('n_ids')}")

    result = eval_1to1(
        cached["embeddings"], cached["ids"],
        impostor_ratio=args.impostor_ratio, seed=args.seed,
    )
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return

    written = write_report(args.out, name, meta, one_to_one=result)
    print(f"  report: {written['json']}")
    rec = result["thresholds"]["recommended"]
    cur = result["at_current_high"]
    print("  ---- summary ----")
    print(f"  n_genuine={result['n_genuine']} n_impostor={result['n_impostor']} "
          f"(ratio={args.impostor_ratio}, seed={args.seed})")
    print(f"  TAR@FAR=0.001 = {result['tar_at_far']:.4f}   (цель ТЗ: >=0.99)")
    print(f"  EER={result['eer']:.4f}  AUC={result['auc']:.4f}")
    print(f"  recommended: high={rec['high']:.4f} low={rec['low']:.4f} "
          f"margin={rec['margin']:.4f} eer={rec['eer']:.4f}")
    print(f"  current(0.60): FAR={cur['far']:.4f} FRR={cur['frr']:.4f} TAR={cur['tar']:.4f}")
    print(f"  FRR@recommended_high={result['frr_at_recommended_high']:.4f} "
          f"(цель ТЗ: FRR<=0.03)")


if __name__ == "__main__":
    main()