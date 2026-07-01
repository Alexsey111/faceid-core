# evaluation/embed_cache.py — кеш эмбеддингов на диске (.npz + .meta.json).
#
# RetinaFace-512 на ~8204 изображениях ≈ 40 мин → кеш обязателен и однократен.
#
# Ключ файла: f"{slug}_det{det_size}" (slug — имя датасета: orig/extracted; det_size —
# размер детектора; 0 для extract-пути без RetinaFace). Файлы:
#   {key}.npz        — embeddings(N,512 float32), ids(N, <U64)), files(N, <U256))
#   {key}.meta.json  — source_root, det_size, model_sha256, n, created_at
# get_or_build проверяет кеш; при отсутствии вызывает builder и сохраняет.

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

CACHE_DIR_DEFAULT = "evaluation/cache"


def cache_key(slug: str, det_size: int) -> str:
    return f"{slug}_det{det_size}"


def cache_paths(cache_dir: str | os.PathLike, slug: str, det_size: int) -> tuple[Path, Path]:
    key = cache_key(slug, det_size)
    base = Path(cache_dir)
    return base / f"{key}.npz", base / f"{key}.meta.json"


def save(
    cache_dir: str | os.PathLike,
    slug: str,
    det_size: int,
    embeddings: np.ndarray,
    ids: list[str],
    files: list[str],
    meta: dict,
) -> tuple[Path, Path]:
    """Сохраняет .npz + .meta.json. Создаёт cache_dir при необходимости."""
    npz_path, meta_path = cache_paths(cache_dir, slug, det_size)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = np.asarray(embeddings, dtype=np.float32)
    ids_arr = np.asarray(ids, dtype="<U64")
    files_arr = np.asarray([str(f) for f in files], dtype="<U256")
    np.savez(npz_path, embeddings=embeddings, ids=ids_arr, files=files_arr)

    full_meta = {
        "slug": slug,
        "det_size": int(det_size),
        "n": int(len(ids)),
        "n_embeddings": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
        **meta,
    }
    meta_path.write_text(json.dumps(full_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return npz_path, meta_path


def load(
    cache_dir: str | os.PathLike, slug: str, det_size: int
) -> dict | None:
    """
    Возвращает {embeddings, ids, files, meta} при наличии валидного кеша, иначе None.
    Валидность: оба файла существуют, .npz содержит embeddings/ids/files одной длины.
    """
    npz_path, meta_path = cache_paths(cache_dir, slug, det_size)
    if not (npz_path.is_file() and meta_path.is_file()):
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            embeddings = data["embeddings"]
            ids = data["ids"]
            files = data["files"]
    except (KeyError, ValueError, OSError):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n = meta.get("n")
    if n is not None and not (len(ids) == n == len(files) == len(embeddings)):
        return None
    return {
        "embeddings": embeddings,
        "ids": [str(s) for s in ids],
        "files": [str(s) for s in files],
        "meta": meta,
    }


def get_or_build(
    cache_dir: str | os.PathLike,
    slug: str,
    det_size: int,
    builder,
    force: bool = False,
) -> dict:
    """
    Возвращает кеш; при отсутствии (или force=True) вызывает builder() →
    (embeddings, ids, files, meta) и сохраняет. builder — вызываемое без аргументов.
    """
    if not force:
        cached = load(cache_dir, slug, det_size)
        if cached is not None:
            return cached
    embeddings, ids, files, meta = builder()
    save(cache_dir, slug, det_size, embeddings, ids, files, meta)
    return load(cache_dir, slug, det_size)  # type: ignore[return-value]