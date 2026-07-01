# evaluation/liveness/dataset.py — сбор списка сэмплов anti-spoofing-датасета.
#
# Структура (Anti-Spoofing Dataset): 5 классов-папок.
#   live_selfie  — .jpg (изображения, живое селфи)
#   live_video   — .mp4 (живое видео)
#   printouts    — .mp4 (атака: распечатка A4)        → attack_type 'print'
#   cut-out printouts — .mp4 (вырезанная распечатка) → attack_type 'cutout'
#   replay       — .mp4/.mov (replay-атака)          → attack_type 'replay'
# Метки: live (live_selfie + live_video) → label 1; attack → label 0.
# Решение пользователя: live = live_selfie + live_video, NPCER по ним вместе.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from evaluation.datasets import IMG_EXTS
from evaluation.liveness.video import VIDEO_EXTS, is_video

logger = logging.getLogger("liveness.dataset")

# class_name → (label, attack_type)
CLASS_MAP = {
    "live_selfie": (1, "live"),
    "live_video": (1, "live"),
    "printouts": (0, "print"),
    "cut-out printouts": (0, "cutout"),
    "replay": (0, "replay"),
}

# Псевдонимы имён папок (на случай различий в регистре/пробелах).
CLASS_ALIASES = {
    "cut-out printouts": "cut-out printouts",
    "cutout printouts": "cut-out printouts",
    "cut-out": "cut-out printouts",
}


def _resolve_class_dir_name(name: str) -> str | None:
    if name in CLASS_MAP:
        return name
    key = name.lower().strip()
    if key in CLASS_MAP:
        return key
    for alias, canon in CLASS_ALIASES.items():
        if alias.lower() == key:
            return canon
    return None


class LivenessSample:
    __slots__ = ("cls", "label", "attack_type", "path", "is_image", "frame_index")

    def __init__(self, cls, label, attack_type, path, is_image, frame_index=None):
        self.cls = cls
        self.label = label
        self.attack_type = attack_type
        self.path = path
        self.is_image = is_image
        self.frame_index = frame_index

    def __repr__(self):
        return (f"LivenessSample(cls={self.cls}, label={self.label}, "
                f"attack_type={self.attack_type}, path={self.path}, "
                f"is_image={self.is_image}, frame_index={self.frame_index})")


def build_liveness_samples(
    root: str | Path,
    n_frames_per_video: int = 30,
) -> list[LivenessSample]:
    """
    Собрать плоский список сэмплов: изображения как есть, видео раскрываются в
    N сэмплов-кадров (frame_index = 0..N-1). Порядок детерминирован (sorted).
    """
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"liveness dataset root not found: {root}")

    samples: list[LivenessSample] = []
    unknown_dirs: list[str] = []

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        canon = _resolve_class_dir_name(entry.name)
        if canon is None:
            unknown_dirs.append(entry.name)
            continue
        label, attack_type = CLASS_MAP[canon]
        is_live = label == 1

        for f in sorted(entry.iterdir()):
            if f.is_dir():
                continue
            suffix = f.suffix.lower()
            if suffix in IMG_EXTS:
                samples.append(LivenessSample(
                    cls=canon, label=label, attack_type=attack_type,
                    path=f, is_image=True, frame_index=None,
                ))
            elif suffix in VIDEO_EXTS:
                for fi in range(n_frames_per_video):
                    samples.append(LivenessSample(
                        cls=canon, label=label, attack_type=attack_type,
                        path=f, is_image=False, frame_index=fi,
                    ))
            # нераспознанные расширения игнорируем (в т.ч. .csv в корне — он не здесь)

    if unknown_dirs:
        logger.warning("unknown class dirs ignored: %s", unknown_dirs)

    return samples


def dataset_stats(samples: list[LivenessSample]) -> dict:
    """Сводка по собранному датасету: счётчики по классу/метке/типу атаки."""
    from collections import Counter
    by_cls = Counter(s.cls for s in samples)
    by_attack = Counter(s.attack_type for s in samples)
    n_live = sum(1 for s in samples if s.label == 1)
    n_attack = sum(1 for s in samples if s.label == 0)
    n_images = sum(1 for s in samples if s.is_image)
    n_video_frames = sum(1 for s in samples if not s.is_image)
    return {
        "n_total": len(samples),
        "n_live": n_live,
        "n_attack": n_attack,
        "n_images": n_images,
        "n_video_frames": n_video_frames,
        "by_class": dict(by_cls),
        "by_attack_type": dict(by_attack),
    }