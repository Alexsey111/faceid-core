# evaluation/extract.py — извлечение ArcFace-эмбеддингов из датасета лиц.
#
# ДВА пути (соответствуют плану P1):
#   orig      — RetinaFace (SCRFD) детекция → 5pt landmarks → norm_crop(112)
#               (повторяет production-пайплайн). Fallback resize(roi,112) при
#               отсутствии landmarks. det_size задаёт размер детектора (512/320).
#   extracted — уже кропнутые 128×128 Haar-лица без landmarks → resize(112)+encode
#               (без выравнивания; ablation, чтобы оценить вклад alignment).
#
# Переиспользует (НЕ дублирует) ML-компоненты app:
#   ImagePreprocessor.process_image / .decode  (app/ml/preprocessing)
#   RetinaFaceDetector(det_size).detect        (app/ml/detection)
#   insightface.utils.face_align.norm_crop     (аффинное выравнивание)
#   OnnxArcFaceEncoder.encode_batch            (app/ml/embedding, напрямую, минуя BatchEncoder)
#
# Модели грузятся ЛЕНИВО (внутри функций) — импорт extract.py не поднимает ONNX.

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from evaluation.datasets import iter_id_images

logger = logging.getLogger("evaluation.extract")

ARCFACE_MODEL_REL = ("buffalo_l", "w600k_r50.onnx")
INPUT_SIZE = (112, 112)


def _read_bgr(path: Path) -> np.ndarray:
    """Читает файл → BGR uint8 (через ImagePreprocessor: decode+resize_if_needed)."""
    from app.ml.preprocessing.image_preprocessor import ImagePreprocessor

    pre = ImagePreprocessor()
    return pre.process(path.read_bytes())


def _crop_roi(image: np.ndarray, bbox: list[float]) -> np.ndarray:
    """Кроп по bbox (xyxy float) с клипингом к границам → BGR roi (для fallback)."""
    h, w = image.shape[:2]
    x1 = max(0, int(round(bbox[0])))
    y1 = max(0, int(round(bbox[1])))
    x2 = min(w, int(round(bbox[2])))
    y2 = min(h, int(round(bbox[3])))
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


def _get_encoder(models_dir: str | os.PathLike):
    """OnnxArcFaceEncoder напрямую (минуя BatchEncoder-timeout-обёртку) для throughput."""
    from app.core.config import settings  # noqa: F401  — гарантирует, что settings загружен
    from app.ml.embedding.onnx_arcface_encoder import OnnxArcFaceEncoder

    model_path = Path(models_dir) / "buffalo_l" / "w600k_r50.onnx"
    if not model_path.is_file():
        raise FileNotFoundError(f"ArcFace model not found: {model_path}")
    return OnnxArcFaceEncoder(model_path)


def _model_sha256(models_dir: str | os.PathLike) -> str:
    path = Path(models_dir) / "buffalo_l" / "w600k_r50.onnx"
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_orig(
    dataset: dict[str, list[Path]],
    det_size: int = 512,
    batch_size: int = 32,
    models_dir: str | os.PathLike | None = None,
    on_progress=None,
) -> tuple[np.ndarray, list[str], list[str], dict]:
    """
    RetinaFace+norm_crop путь. Возвращает (embeddings(N,512), ids, files, stats).
    Изображения без детекции пропускаются (stats['n_no_face']).
    """
    from app.ml.detection.retinaface_detector import RetinaFaceDetector
    from insightface.utils.face_align import norm_crop

    models_dir = models_dir or _default_models_dir()
    detector = RetinaFaceDetector(det_size)
    encoder = _get_encoder(models_dir)

    crops: list[np.ndarray] = []
    ids: list[str] = []
    files: list[str] = []
    n_aligned = 0
    n_fallback = 0
    n_no_face = 0
    n_total = sum(len(v) for v in dataset.values())
    done = 0

    for id_, path in iter_id_images(dataset):
        try:
            image = _read_bgr(path)
        except Exception as exc:
            logger.warning("decode failed %s: %s", path, exc)
            n_no_face += 1
            done += 1
            continue
        faces = detector.detect(image)
        if not faces:
            n_no_face += 1
            done += 1
            if on_progress and done % 100 == 0:
                on_progress(done, n_total)
            continue
        top = faces[0]
        landmarks = top.get("landmarks")
        if landmarks is not None:
            face_input = norm_crop(image, np.asarray(landmarks, dtype=np.float32), 112)
            n_aligned += 1
        else:
            roi = _crop_roi(image, top["bbox"])
            face_input = cv2.resize(roi, INPUT_SIZE)
            n_fallback += 1
        crops.append(face_input)
        ids.append(id_)
        files.append(str(path))
        done += 1
        if on_progress and done % 100 == 0:
            on_progress(done, n_total)
        # батч-кодирование по мере накопления
        if len(crops) >= batch_size:
            _flush(crops, ids, files, encoder)
    if crops:
        _flush(crops, ids, files, encoder)

    emb, ids_out, files_out = _consume_flushed()
    stats = {
        "n_total": n_total,
        "n_encoded": len(ids_out),
        "n_aligned": n_aligned,
        "n_fallback": n_fallback,
        "n_no_face": n_no_face,
        "det_size": int(det_size),
        "model_sha256": _model_sha256(models_dir),
        "path": "orig:retinaface+norm_crop(112)",
    }
    return emb, ids_out, files_out, stats


def extract_extracted(
    dataset: dict[str, list[Path]],
    batch_size: int = 32,
    models_dir: str | os.PathLike | None = None,
    on_progress=None,
) -> tuple[np.ndarray, list[str], list[str], dict]:
    """
    Путь для уже кропнутых лиц (128×128 Haar): resize→112+encode, без детекции/выравнивания.
    """
    models_dir = models_dir or _default_models_dir()
    encoder = _get_encoder(models_dir)

    crops: list[np.ndarray] = []
    ids: list[str] = []
    files: list[str] = []
    n_total = sum(len(v) for v in dataset.values())
    done = 0
    n_decode_fail = 0

    for id_, path in iter_id_images(dataset):
        try:
            image = _read_bgr(path)
        except Exception as exc:
            logger.warning("decode failed %s: %s", path, exc)
            n_decode_fail += 1
            done += 1
            continue
        crops.append(cv2.resize(image, INPUT_SIZE))
        ids.append(id_)
        files.append(str(path))
        done += 1
        if on_progress and done % 100 == 0:
            on_progress(done, n_total)
        if len(crops) >= batch_size:
            _flush(crops, ids, files, encoder)
    if crops:
        _flush(crops, ids, files, encoder)

    emb, ids_out, files_out = _consume_flushed()
    stats = {
        "n_total": n_total,
        "n_encoded": len(ids_out),
        "n_decode_fail": n_decode_fail,
        "det_size": 0,
        "model_sha256": _model_sha256(models_dir),
        "path": "extracted:resize(112)+encode",
    }
    return emb, ids_out, files_out, stats


# --- внутренний «продуваемый» буфер: кодируем батчами, склеиваем в конце -------
_FLUSH_EMB: list[np.ndarray] = []
_FLUSH_IDS: list[str] = []
_FLUSH_FILES: list[str] = []


def _flush(crops, ids, files, encoder):
    """Кодирует текущий батч, дописывает в накопитель и ОЧИЩАЕТ все три буфера.

    Очистка ids/files обязательна: иначе при многократных батчах _FLUSH_IDS/_FILES
    росли бы геометрически (extend добавлял бы весь накопленный к текущему моменту
    список заново на каждом батче). crops очищался вручную, а ids/files — нет (баг).
    """
    emb = encoder.encode_batch(crops)
    _FLUSH_EMB.append(np.asarray(emb, dtype=np.float32))
    _FLUSH_IDS.extend(ids)
    _FLUSH_FILES.extend(files)
    crops.clear()
    ids.clear()
    files.clear()


def _consume_flushed():
    if _FLUSH_EMB:
        emb = np.concatenate(_FLUSH_EMB, axis=0)
    else:
        emb = np.empty((0, 512), dtype=np.float32)
    ids_out = list(_FLUSH_IDS)
    files_out = list(_FLUSH_FILES)
    _FLUSH_EMB.clear()
    _FLUSH_IDS.clear()
    _FLUSH_FILES.clear()
    return emb, ids_out, files_out


def _default_models_dir() -> str:
    """
    Каталог, НАПРЯМУЮ содержащий buffalo_l/w600k_r50.onnx (для энкодера).
    Детектор (FaceAnalysis) использует другую логику (_detect_models_root →
    родитель 'models'), но энкодер OnnxArcFaceEncoder ждёт путь именно к файлу
    <models_dir>/buffalo_l/w600k_r50.onnx. Ищем среди кандидатов.
    """
    from pathlib import Path

    from app.core.config import settings

    from app.ml.runtime import PROJECT_ROOT

    candidates = [
        Path(settings.MODELS_DIR),
        PROJECT_ROOT / "models",
        PROJECT_ROOT.parent / "models",
    ]
    for cand in candidates:
        if (cand / "buffalo_l" / "w600k_r50.onnx").is_file():
            return str(cand)
    # fallback — пусть encoder честно упадёт с понятной ошибкой про путь.
    return str(Path(settings.MODELS_DIR))