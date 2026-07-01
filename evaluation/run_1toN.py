# evaluation/run_1toN.py — CLI 1:N identification (rank-1/rank-5/CMC).
#
#   python -m evaluation.run_1toN --dataset {orig|extracted}
#       [--cache-dir evaluation/cache] [--det-size 512] [--max-rank 50]
#       [--validate-faiss] [--out evaluation/out] [--report-name NAME]
#
# Gallery = первый сэмпл каждого id, probe = остальные. --validate-faiss сверяет с FaissIndex.

from __future__ import annotations

import argparse

from evaluation.cli_common import (
    add_dataset_arg,
    cache_dir_arg,
    dataset_meta_from_cache,
    det_size_for,
    load_cache,
)
from evaluation.protocols import eval_1toN, validate_faiss_consistency
from evaluation.report import OUT_DIR_DEFAULT, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="1:N face identification eval (CMC)")
    add_dataset_arg(parser)
    cache_dir_arg(parser)
    parser.add_argument("--det-size", type=int, default=512, help="должен совпадать с кешем orig")
    parser.add_argument("--max-rank", type=int, default=50)
    parser.add_argument("--validate-faiss", action="store_true", help="сверка numpy vs FaissIndex")
    parser.add_argument("--out", default=OUT_DIR_DEFAULT)
    parser.add_argument("--report-name", default=None)
    args = parser.parse_args()

    det_size = det_size_for(args.dataset, args.det_size)
    name = args.report_name or f"{args.dataset}_1toN"

    cached = load_cache(args.cache_dir, args.dataset, det_size)
    meta = dataset_meta_from_cache(cached)
    print(f"[run_1toN] {name}: n_images={meta.get('n_images')} n_ids={meta.get('n_ids')}")

    result = eval_1toN(cached["embeddings"], cached["ids"], max_rank=args.max_rank)
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return

    faiss_check = None
    if args.validate_faiss:
        print("  [validate_faiss] сверка numpy vs FaissIndex...")
        faiss_check = validate_faiss_consistency(
            cached["embeddings"], cached["ids"], max_rank=min(args.max_rank, 5)
        )
        if "error" in faiss_check:
            print(f"    faiss check ERROR: {faiss_check['error']}")
            faiss_check = None
        else:
            print(f"    mismatch={faiss_check['mismatch']}/{faiss_check['n_probes']}")

    written = write_report(args.out, name, meta, one_to_n=result, faiss_check=faiss_check)
    print(f"  report: {written['json']}")
    print("  ---- summary ----")
    print(f"  n_gallery={result['n_gallery']} n_probes={result['n_probes']} "
          f"n_ids_single_image={result['n_ids_single_image']}")
    print(f"  rank-1={result['rank1']:.4f}  rank-5={result['rank5']:.4f}  (max_rank={args.max_rank})")
    if faiss_check and "error" not in faiss_check:
        print(f"  faiss_vs_numpy_mismatch={faiss_check['mismatch']}")


if __name__ == "__main__":
    main()