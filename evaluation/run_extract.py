# evaluation/run_extract.py — CLI построения кеша эмбеддингов.
#
#   python -m evaluation.run_extract --dataset {orig|extracted}
#       [--root PATH] [--cache-dir evaluation/cache] [--det-size 512]
#       [--batch 32] [--models-dir PATH] [--force]
#
# orig: RetinaFace(det_size)+norm_crop+ArcFace. extracted: resize(112)+ArcFace (det_size=0).
# Кеш: {slug}_det{det_size}.npz + .meta.json. RetinaFace-512 на ~8204 imgs ≈ 40 мин (однократно).

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from evaluation.cli_common import (
    add_dataset_arg,
    cache_dir_arg,
    det_size_for,
    ensure_models_env,
    resolve_root,
)
from evaluation.datasets import dataset_stats, load_folder_dataset
from evaluation.embed_cache import get_or_build


def _progress(done, total):
    if total:
        pct = 100.0 * done / total
        print(f"  extract {done}/{total} ({pct:.1f}%)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build embedding cache for eval-harness")
    add_dataset_arg(parser)
    cache_dir_arg(parser)
    parser.add_argument("--det-size", type=int, default=512, help="RetinaFace det_size (orig)")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--force", action="store_true", help="перестроить кеш")
    args = parser.parse_args()

    ensure_models_env()
    root = resolve_root(args)
    det_size = det_size_for(args.dataset, args.det_size)

    print(f"[run_extract] dataset={args.dataset} root={root} det_size={det_size} batch={args.batch}")
    dataset = load_folder_dataset(root)
    stats = dataset_stats(dataset)
    print(f"  dataset: {stats}")

    def builder():
        if args.dataset == "orig":
            from evaluation.extract import extract_orig
            t0 = time.time()
            emb, ids, files, ext_stats = extract_orig(
                dataset, det_size=det_size, batch_size=args.batch,
                models_dir=args.models_dir, on_progress=_progress,
            )
        else:
            from evaluation.extract import extract_extracted
            t0 = time.time()
            emb, ids, files, ext_stats = extract_extracted(
                dataset, batch_size=args.batch, models_dir=args.models_dir,
                on_progress=_progress,
            )
        ext_stats["elapsed_s"] = round(time.time() - t0, 1)
        ext_stats["source_root"] = str(root)
        ext_stats["created_at"] = datetime.now(timezone.utc).isoformat()
        return emb, ids, files, ext_stats

    cached = get_or_build(args.cache_dir, args.dataset, det_size, builder, force=args.force)
    meta = cached["meta"]
    print(f"[run_extract] DONE. cache: {args.dataset}_det{det_size}")
    print(f"  n_embeddings={len(cached['ids'])} dim={cached['embeddings'].shape[-1]}")
    print(f"  path={meta.get('path')} n_aligned={meta.get('n_aligned')} "
          f"n_fallback={meta.get('n_fallback')} n_no_face={meta.get('n_no_face')}")


if __name__ == "__main__":
    main()