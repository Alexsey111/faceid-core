# evaluation/datasets.py — загрузка папочного датасета лиц в детерминированную структуру.
#
# Контракт: root указывает НА ПАПКУ, содержащую подпапки-идентичности
#   <root>/<id>/<n>.jpg   (один уровень id-папок, внутри — изображения).
# Для CelebData корневая папка имеет обёртку («Face Data/Face Dataset»),
# поэтому CLI передаёт именно внутренний root (см. run_extract.DATASET_ROOTS).
#
# Возвращает dict {id: [sorted Path]}, отсортированный по id; файлы внутри — sorted.
# Файлы без распознанного расширения игнорируются.

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_folder_dataset(root: str | os.PathLike) -> dict[str, list[Path]]:
    """
    Сканирует root: каждая прямая поддиректория = identity (её имя = id),
    изображения внутри (по расширению) = сэмплы. Возвращает {id: [sorted Path]}.
    Папки без изображений пропускаются.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")

    dataset: dict[str, list[Path]] = {}
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        imgs = sorted(
            (f for f in entry.iterdir() if f.suffix.lower() in IMG_EXTS),
            key=lambda p: p.name,
        )
        if imgs:
            dataset[entry.name] = imgs
    return dataset


def iter_id_images(dataset: dict[str, list[Path]]) -> Iterator[tuple[str, Path]]:
    """
    Плоский детерминированный итератор (id, path) — sorted по id, затем по файлу.
    Этот порядок задаёт индексацию сэмплов в массиве эмбеддингов.
    """
    for id_ in sorted(dataset.keys()):
        for path in dataset[id_]:
            yield id_, path


def flatten(dataset: dict[str, list[Path]]) -> tuple[list[Path], list[str]]:
    """
    (files, ids) — параллельные списки в порядке iter_id_images.
    Удобно для построения массива эмбеддингов и ids-массива.
    """
    files: list[Path] = []
    ids: list[str] = []
    for id_, path in iter_id_images(dataset):
        files.append(path)
        ids.append(id_)
    return files, ids


def dataset_stats(dataset: dict[str, list[Path]]) -> dict:
    """Сводка по датасету для отчёта: n_ids, n_images, per-id min/mean/max."""
    counts = [len(v) for v in dataset.values()]
    n_ids = len(dataset)
    n_images = sum(counts)
    return {
        "n_ids": n_ids,
        "n_images": n_images,
        "min_per_id": min(counts) if counts else 0,
        "mean_per_id": (n_images / n_ids) if n_ids else 0.0,
        "max_per_id": max(counts) if counts else 0,
        "n_single_image_ids": sum(1 for c in counts if c == 1),
    }


def iter_id_images_typed(dataset: Iterable[tuple[str, Path]]) -> Iterator[tuple[str, Path]]:
    """Совместимость: если передан уже плоский итерируемый (id, path) — пропускает."""
    for id_, path in dataset:  # type: ignore[misc]
        yield id_, path