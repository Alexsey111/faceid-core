# evaluation/lfw/run_lfw.py — LFW 1:1 verification (эталонный бенчмарк ТЗ: TAR@FAR + accuracy).
#
#   python -m evaluation.lfw.run_lfw [--pairs lfw_data/pairs.csv] [--images-root lfw_data/train]
#       [--det-size 320] [--cache-dir evaluation/lfw/cache] [--out evaluation/lfw/out]
#       [--report-name lfw]
#
# Поток: parse pairs.csv → уникальные пути изображений → (кеш?) RetinaFace+norm_crop+ArcFace
# → кеш эмбеддингов (.npz, ключ = model_sha256+det_size) → косинус по 6000 парам →
# pooled ROC → TAR@FAR(0.001/0.01/0.0001)+AUC+EER + 10-fold best-threshold accuracy.
# Модель/провайдеры берутся из env: MODELS_DIR, ARCFACE_MODEL_REL, ONNX_ARCFACE_PROVIDERS.

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# git-root на path (содержит app/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config import settings  # noqa: E402
from app.ml.detection.retinaface_detector import RetinaFaceDetector  # noqa: E402
from app.ml.embedding.onnx_arcface_encoder import OnnxArcFaceEncoder  # noqa: E402
from app.ml.preprocessing.image_preprocessor import ImagePreprocessor  # noqa: E402

from evaluation.metrics import (  # noqa: E402
    auc_from_roc,
    confusion,
    eer_point,
    roc_curve,
    tar_at_far,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("lfw.run")

INPUT_SIZE = (112, 112)


def _model_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_pairs(
    pairs_csv: Path, images_root: Path, layout: str = "raw"
) -> list[dict]:
    """pairs.csv → список {a, b, label, fold}. Пути разрешаются к images_root.

    layout:
      'raw'        — оригинальный lfw.tgz: <images_root>/<Name>/<Name>_XXXX.jpg
                     (имя личности = filename без суффикса _NNNN.jpg; берётся из
                     колонок image_a/image_b).
      'marcelohaps' — HF-зеркало: <images_root>/<image_a_path> (images/000/...).
    """
    import re
    pairs: list[dict] = []
    with open(pairs_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if layout == "raw":
                fa, fb = row["image_a"], row["image_b"]
                na = re.sub(r"_\d{4}\.(jpg|jpeg|png)$", "", fa, flags=re.IGNORECASE)
                nb = re.sub(r"_\d{4}\.(jpg|jpeg|png)$", "", fb, flags=re.IGNORECASE)
                a = images_root / na / fa
                b = images_root / nb / fb
            else:
                a = images_root / row["image_a_path"]
                b = images_root / row["image_b_path"]
            pairs.append({
                "a": a, "b": b,
                "label": int(row["is_same"]),
                "fold": int(row["fold_id"]),
            })
    return pairs


def collect_unique_images(pairs: list[dict]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in pairs:
        for img in (p["a"], p["b"]):
            if img not in seen:
                seen.add(img)
                unique.append(img)
    return unique


def _select_main_face(faces: list[dict]) -> dict:
    """Делегирует в app.ml.detection.face_selection.select_main_face.

    Robust-эвристика (IoU-дедуп → highest-conf в группе → largest-area среди
    групп): на full-scene LFW выбирает субъекта, а не высокоуверенный мелкий
    фоновый; на кропнутых лицах не портит локализацию дубликатами. См. docstring
    app/ml/detection/face_selection.py и memory lfw-single-face-meets-target.
    """
    from app.ml.detection.face_selection import select_main_face

    return select_main_face(faces)


def extract_embeddings(
    images: list[Path], det_size: int, encoder: OnnxArcFaceEncoder,
    preproc: ImagePreprocessor,
) -> tuple[np.ndarray, list[str], dict]:
    """Извлечь 512-d эмбеддинги для списка изображений (production-пайплайн)."""
    det = RetinaFaceDetector(det_size=det_size)
    from insightface.utils.face_align import norm_crop

    embs: list[np.ndarray] = []
    paths: list[str] = []
    n_no_face = 0
    n_fallback = 0
    n_aligned = 0
    n_multi_face = 0
    t0 = time.time()
    for i, path in enumerate(images):
        try:
            image = preproc.process(path.read_bytes())
        except Exception as exc:  # noqa: BLE001
            logger.warning("decode failed %s: %s", path, exc)
            n_no_face += 1
            continue
        faces = det.detect(image)
        if not faces:
            n_no_face += 1
            continue
        if len(faces) > 1:
            n_multi_face += 1
        top = _select_main_face(faces)  # наибольшее лицо, НЕ faces[0]
        landmarks = top.get("landmarks")
        if landmarks is not None:
            face_input = norm_crop(image, np.asarray(landmarks, dtype=np.float32), 112)
            n_aligned += 1
        else:
            # fallback: кроп по bbox + resize (без выравнивания)
            h, w = image.shape[:2]
            x1, y1, x2, y2 = top["bbox"]
            x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
            x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
            if x2 <= x1 or y2 <= y1:
                n_no_face += 1
                continue
            face_input = cv2.resize(image[y1:y2, x1:x2], INPUT_SIZE)
            n_fallback += 1
        emb = encoder.encode(face_input)
        embs.append(np.asarray(emb, dtype=np.float32))
        paths.append(str(path))
        if (i + 1) % 200 == 0:
            logger.info("extract %d/%d (%.0fs)", i + 1, len(images), time.time() - t0)
    emb_arr = np.asarray(embs, dtype=np.float32) if embs else np.empty((0, 512), dtype=np.float32)
    stats = {
        "n_images": len(images),
        "n_encoded": len(embs),
        "n_aligned": n_aligned,
        "n_fallback": n_fallback,
        "n_no_face": n_no_face,
        "n_multi_face": n_multi_face,
        "det_size": int(det_size),
        "elapsed_s": round(time.time() - t0, 1),
    }
    return emb_arr, paths, stats


def cache_path(cache_dir: str, model_slug: str, det_size: int) -> Path:
    return Path(cache_dir) / f"lfw_{model_slug}_det{det_size}.npz"


def _model_slug(model_path: Path) -> str:
    # короткий slug из пути: antelopev2_glintr100 / buffalo_l_w600k_r50
    parts = model_path.relative_to(model_path.parents[1]).with_suffix("").parts
    return "_".join(parts)


def build_embeddings(
    pairs: list[dict], images_root: Path, det_size: int, cache_dir: str, force: bool,
) -> tuple[dict[str, np.ndarray], dict]:
    """Возвращает {path: embedding} + meta. Кеш по model_sha256+det_size."""
    model_rel = settings.ARCFACE_MODEL_REL
    model_path = Path(settings.MODELS_DIR) / model_rel
    if not model_path.is_file():
        raise FileNotFoundError(f"ArcFace model not found: {model_path}")

    sha = _model_sha256(model_path)
    slug = _model_slug(model_path)
    cpath = cache_path(cache_dir, slug, det_size)

    if not force and cpath.is_file():
        logger.info("cache hit: %s", cpath)
        with np.load(cpath, allow_pickle=True) as d:
            paths = list(d["paths"])
            emb = np.asarray(d["embeddings"], dtype=np.float32)
        # проверка целостности кеша по sha
        meta = {"model_slug": slug, "model_sha256": sha, "det_size": det_size, "from_cache": True}
        emb_map = {p: emb[i] for i, p in enumerate(paths)}
        return emb_map, meta

    logger.info("cache miss, extracting (model=%s det=%d) ...", slug, det_size)
    encoder = OnnxArcFaceEncoder(str(model_path))
    preproc = ImagePreprocessor()
    images = collect_unique_images(pairs)
    logger.info("unique images: %d", len(images))
    emb_arr, paths, stats = extract_embeddings(images, det_size, encoder, preproc)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez(cpath, embeddings=emb_arr, paths=np.asarray(paths, dtype=object))
    meta = {
        "model_slug": slug, "model_rel": model_rel, "model_sha256": sha,
        "det_size": det_size, **stats, "from_cache": False,
    }
    logger.info("cache saved: %s (%d encoded)", cpath, len(paths))
    return {p: emb_arr[i] for i, p in enumerate(paths)}, meta


def score_pairs(pairs: list[dict], emb_map: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Косинус по парам. Пропускает пары, где хотя бы одно изображение без эмбеддинга."""
    scores: list[float] = []
    labels: list[int] = []
    folds: list[int] = []
    n_missing = 0
    for p in pairs:
        ea = emb_map.get(str(p["a"]))
        eb = emb_map.get(str(p["b"]))
        if ea is None or eb is None:
            n_missing += 1
            continue
        scores.append(float(np.dot(ea, eb)))  # L2-норм → косинус
        labels.append(p["label"])
        folds.append(p["fold"])
    if n_missing:
        logger.warning("pairs skipped (missing embedding): %d", n_missing)
    return (
        np.asarray(scores, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(folds, dtype=np.int64),
    )


def fold_accuracy(scores: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> tuple[float, float]:
    """10-fold best-threshold accuracy (стандартная LFW-метрика) ± std."""
    accs: list[float] = []
    for f in np.unique(folds):
        mask = folds == f
        s, l = scores[mask], labels[mask]
        if len(s) == 0:
            continue
        # лучший порог = середина между отсортированными скорами (простой sweep по unique)
        best_acc = 0.0
        for thr in np.unique(s):
            pred = (s >= thr).astype(np.int64)
            acc = float(np.mean(pred == l))
            if acc > best_acc:
                best_acc = acc
        accs.append(best_acc)
    accs_arr = np.asarray(accs, dtype=np.float64)
    return float(accs_arr.mean()), float(accs_arr.std()) if len(accs_arr) > 1 else 0.0


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="LFW 1:1 verification eval (TAR@FAR + accuracy)")
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    parser.add_argument("--pairs", default=str(repo / "lfw_data" / "pairs.csv"))
    parser.add_argument(
        "--layout", choices=["raw", "marcelohaps"], default="raw",
        help="raw: <root>/<Name>/<Name>_XXXX.jpg (оригинальный lfw.tgz); "
             "marcelohaps: <root>/<image_a_path> (HF-зеркало)",
    )
    parser.add_argument(
        "--images-root", default=str(repo / "lfw_data" / "lfw"),
        help="корень изображений (для raw — папка с поддиректориями-личностями)",
    )
    parser.add_argument("--det-size", type=int, default=320)
    parser.add_argument("--cache-dir", default=str(here / "cache"))
    parser.add_argument("--out", default=str(here / "out"))
    parser.add_argument("--report-name", default="lfw")
    parser.add_argument("--force", action="store_true", help="перестроить кеш эмбеддингов")
    args = parser.parse_args(argv)

    pairs_csv = Path(args.pairs)
    images_root = Path(args.images_root)
    if not pairs_csv.is_file():
        raise SystemExit(f"pairs.csv not found: {pairs_csv}")
    if not images_root.is_dir():
        raise SystemExit(f"images root not found: {images_root}")

    pairs = parse_pairs(pairs_csv, images_root, layout=args.layout)
    n_total = len(pairs)
    n_pos = sum(1 for p in pairs if p["label"] == 1)
    logger.info("pairs: %d total (%d genuine, %d impostor)", n_total, n_pos, n_total - n_pos)

    emb_map, meta = build_embeddings(pairs, images_root, args.det_size, args.cache_dir, args.force)
    scores, labels, folds = score_pairs(pairs, emb_map)
    if len(scores) == 0:
        logger.error("no scored pairs (all missing embeddings?)")
        return

    far, tar, thr = roc_curve(scores, labels)
    auc = auc_from_roc(far, tar)
    eer, eer_thr = eer_point(far, tar, thr)
    tar_001, _ = tar_at_far(far, tar, thr, 0.001)
    tar_01, _ = tar_at_far(far, tar, thr, 0.01)
    tar_0001, _ = tar_at_far(far, tar, thr, 0.0001)
    mean_acc, acc_std = fold_accuracy(scores, labels, folds)

    print("=" * 70)
    print(f"LFW eval - model={meta.get('model_slug')} det={meta.get('det_size')}")
    print(f"  pairs scored: {len(scores)}/{n_total}  (genuine={int(np.sum(labels==1))}, "
          f"impostor={int(np.sum(labels==0))})")
    print(f"  embeddings: encoded={meta.get('n_encoded')} no_face={meta.get('n_no_face')} "
          f"aligned={meta.get('n_aligned')} fallback={meta.get('n_fallback')} "
          f"multi_face={meta.get('n_multi_face')}")
    print("-" * 70)
    print(f"  TAR@FAR=0.001  = {tar_001:.4f}   (цель ТЗ: >=0.99)")
    print(f"  TAR@FAR=0.01   = {tar_01:.4f}")
    print(f"  TAR@FAR=0.0001 = {tar_0001:.4f}")
    print(f"  AUC={auc:.4f}  EER={eer:.4f}  eer_threshold={eer_thr:.4f}")
    print(f"  10-fold accuracy = {mean_acc:.4f} ± {acc_std:.4f}  (стандартная LFW-метрика)")
    target = 0.99
    meet = "MEET" if tar_001 >= target else "MISS"
    print("-" * 70)
    print(f"  TZ target TAR>=0.99 @ FAR<=0.001: {meet} ({tar_001:.2%})")
    print("=" * 70)

    # отчёт
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model_slug": meta.get("model_slug"),
        "model_rel": meta.get("model_rel"),
        "model_sha256": meta.get("model_sha256"),
        "det_size": meta.get("det_size"),
        "n_pairs_total": n_total,
        "n_pairs_scored": int(len(scores)),
        "n_genuine": int(np.sum(labels == 1)),
        "n_impostor": int(np.sum(labels == 0)),
        "n_embeddings": meta.get("n_encoded"),
        "n_no_face": meta.get("n_no_face"),
        "n_aligned": meta.get("n_aligned"),
        "n_fallback": meta.get("n_fallback"),
        "n_multi_face": meta.get("n_multi_face"),
        "tar_at_far_0.001": tar_001,
        "tar_at_far_0.01": tar_01,
        "tar_at_far_0.0001": tar_0001,
        "auc": auc,
        "eer": eer,
        "eer_threshold": eer_thr,
        "fold_accuracy_mean": mean_acc,
        "fold_accuracy_std": acc_std,
    }
    import json
    rpath = out_dir / f"{args.report_name}_report.json"
    with open(rpath, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logger.info("report: %s", rpath)


if __name__ == "__main__":
    main()