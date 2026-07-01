# evaluation/cli_common.py — общие хелперы для CLI run_extract/run_1to1/run_1toN.

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from evaluation.embed_cache import CACHE_DIR_DEFAULT, load

# Дефолтные корни датасетов (вложенные обёртки уже раскрыты → папка с <id>/<n>.jpg).
REPO_ROOT = Path(r"C:/Users/Worker/ai/faceid-core")
DATASET_ROOTS = {
    "orig": REPO_ROOT / "Face Data" / "Face Dataset",
    "extracted": REPO_ROOT / "Extracted Faces" / "Extracted Faces",
}
# slug == dataset name (orig/extracted); det_size различает кеши.
SLUGS = ("orig", "extracted")


def add_dataset_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=SLUGS, required=True, help="какой датасет")
    parser.add_argument("--root", default=None, help="переопределить корень датасета")


def resolve_root(args) -> Path:
    root = Path(args.root) if args.root else DATASET_ROOTS[args.dataset]
    if not root.is_dir():
        raise SystemExit(f"dataset root not found: {root}")
    return root


def cache_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)


def load_cache(cache_dir: str, dataset: str, det_size: int) -> dict:
    cached = load(cache_dir, dataset, det_size)
    if cached is None:
        raise SystemExit(
            f"cache miss: {dataset} det{det_size}. "
            f"Сначала запустите `python -m evaluation.run_extract --dataset {dataset}"
            + (f" --det-size {det_size}" if det_size else "")
            + "`."
        )
    return cached


def dataset_meta_from_cache(cached: dict) -> dict:
    """Сводка по датасету/екстрации из кеша (без volatile-полей — для детерминизма отчёта)."""
    meta = dict(cached.get("meta", {}))
    ids = cached["ids"]
    emb = np.asarray(cached["embeddings"])
    _unique = np.unique(np.asarray(ids))
    # удаляем volatile-поля (created_at, elapsed_s) — отчёт должен быть байт-идентичен
    # при повторном прогоне run_1to1/run_1toN на том же кеше.
    meta.pop("created_at", None)
    meta.pop("elapsed_s", None)
    meta["n_ids"] = int(len(_unique))
    meta["n_images"] = int(len(ids))
    meta["embedding_dim"] = int(emb.shape[1]) if emb.ndim == 2 else None
    return meta


def det_size_for(dataset: str, det_size: int | None) -> int:
    # extracted-путь не использует детектор → det_size=0 (соглашение о ключе кеша).
    if dataset == "extracted":
        return 0
    return int(det_size if det_size is not None else 512)


def ensure_models_env() -> None:
    """MODELS_DIR по умолчанию — repo-local models/, если не задан явно."""
    if not os.environ.get("MODELS_DIR"):
        cand = REPO_ROOT / "models"
        if (cand / "buffalo_l" / "w600k_r50.onnx").is_file():
            os.environ["MODELS_DIR"] = str(cand)