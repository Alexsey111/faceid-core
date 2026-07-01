# evaluation/liveness/video.py — равномерный сэмплинг кадров из видео.
#
# cv2.VideoCapture + cap.set(CAP_PROP_POS_FRAMES) по индексам
# round(i*(total-1)/(N-1)). .mov может не открыться без FFmpeg-backend →
# gracefully skip (вызывает callback open_failed). Каждый кадр — BGR ndarray,
# совместимый с ImagePreprocessor.process_image.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger("liveness.video")

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def _uniform_indices(total: int, n_frames: int) -> list[int]:
    """Индексы кадров, равномерно распределённые по таймлайну [0, total-1]."""
    if total <= 0:
        return []
    if n_frames <= 0:
        return []
    if total == 1 or n_frames == 1:
        return [0]
    n = min(n_frames, total)
    return [round(i * (total - 1) / (n - 1)) for i in range(n)]


def iter_video_frames(
    path: str | Path,
    n_frames: int = 30,
    max_side: int | None = None,
) -> Iterator[np.ndarray]:
    """
    Yield до n_frames BGR-кадров, равномерно сэмплированных по таймлайну видео.

    Yields BGR ndarray (HxWx3, uint8). Если видео не открылось (нет backend для
    .mov) или frame_count неизвестен — итератор пуст (caller считает open_fail).
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.warning("video not opened: %s", path)
        return

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            # fallback: пробуем читать подряд, пока не соберём n_frames или поток не кончится.
            count = 0
            while count < n_frames:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if max_side is not None:
                    frame = _maybe_downscale(frame, max_side)
                yield frame
                count += 1
            return

        for idx in _uniform_indices(total, n_frames):
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, idx):
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if max_side is not None:
                frame = _maybe_downscale(frame, max_side)
            yield frame
    finally:
        cap.release()


def _maybe_downscale(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return frame
    scale = max_side / float(s)
    return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def count_frames(path: str | Path, n_frames: int = 30) -> int:
    """Сколько кадров реально будет сэмплено (для планирования без открытия потока)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return 0
        return min(n_frames, total)
    finally:
        cap.release()


def read_first_frame(path: str | Path) -> Optional[np.ndarray]:
    """Первый кадр видео (для smoke-проверки backend). None если не открылось."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        ok, frame = cap.read()
        return frame if (ok and frame is not None) else None
    finally:
        cap.release()