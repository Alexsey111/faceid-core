# evaluation/lfw/run_single_face_eval.py — LFW single-face subset eval.
#
# Подтверждение/опровержение memory lfw-single-face-meets-target: утверждение что
# TAR@FAR=0.001 на LFW single-face subset (оба фото — ровно 1 лицо) = 0.9964, тогда
# как ALL-pairs = 0.9533 (многолицые фото → ошибка выбора лица → битый эмбеддинг).
#
# Переиспользует кеш эмбеддингов run_lfw.py (paths→embedding) + отдельный кеш
# face-counts (прогон детекции без encode). Не пересчитывает эмбеддинги.
#
# Считает TAR@FAR(0.001/0.01) + AUC + EER для трёх срезов:
#   ALL            — все 6000 пар (должно совпасть с lfw_w600k_r50_report 0.9533);
#   SINGLE-SINGLE  — оба фото ровно 1 лицо (memory: 3927 пар → 0.9964);
#   MULTI-INVOLVED — хотя бы одно фото multi-face (контроль: где теряется точность).
#
#   python -m evaluation.lfw.run_single_face_eval [--det-size 320] [--force-counts]

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # git-root on path

from app.core.config import settings  # noqa: E402
from app.ml.detection.retinaface_detector import RetinaFaceDetector  # noqa: E402
from app.ml.preprocessing.image_preprocessor import ImagePreprocessor  # noqa: E402

from evaluation.lfw.run_lfw import (  # noqa: E402
    cache_path,
    parse_pairs,
    _model_slug,
)
from evaluation.metrics import (  # noqa: E402
    auc_from_roc,
    eer_point,
    roc_curve,
    tar_at_far,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("lfw.single_face")


def _model_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def counts_cache_path(cache_dir: str, slug: str, det_size: int) -> Path:
    return Path(cache_dir) / f"lfw_{slug}_det{det_size}_facecounts.npz"


def compute_face_counts(
    paths: list[str], det_size: int, cache_dir: str, slug: str, force: bool,
) -> dict[str, int]:
    """Прогон детектора → {path: n_faces}. Кеш в facecounts.npz (отдельно от embeddings)."""
    cpath = counts_cache_path(cache_dir, slug, det_size)
    if not force and cpath.is_file():
        logger.info("counts cache hit: %s", cpath)
        with np.load(cpath, allow_pickle=True) as d:
            p = list(d["paths"])
            n = list(d["n_faces"])
        return {str(p[i]): int(n[i]) for i in range(len(p))}

    logger.info("computing face counts (det=%d) for %d images ...", det_size, len(paths))
    det = RetinaFaceDetector(det_size=det_size)
    preproc = ImagePreprocessor()
    counts: dict[str, int] = {}
    t0 = time.time()
    for i, p in enumerate(paths):
        try:
            image = preproc.process(Path(p).read_bytes())
            faces = det.detect(image)
            counts[p] = len(faces) if faces else 0
        except Exception:
            counts[p] = 0
        if (i + 1) % 500 == 0:
            logger.info("detect %d/%d (%.0fs)", i + 1, len(paths), time.time() - t0)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez(cpath, paths=np.asarray(list(counts.keys()), dtype=object),
             n_faces=np.asarray(list(counts.values()), dtype=np.int64))
    logger.info("counts cache saved: %s (%.0fs)", cpath, time.time() - t0)
    return counts


def _subset_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    if len(scores) == 0 or (np.sum(labels == 1) == 0) or (np.sum(labels == 0) == 0):
        return {"n_pairs": int(len(scores)), "n_genuine": int(np.sum(labels == 1)),
                "n_impostor": int(np.sum(labels == 0)), "error": "need both classes"}
    far, tar, thr = roc_curve(scores, labels)
    auc = auc_from_roc(far, tar)
    eer, eer_thr = eer_point(far, tar, thr)
    tar_001, _ = tar_at_far(far, tar, thr, 0.001)
    tar_01, _ = tar_at_far(far, tar, thr, 0.01)
    return {
        "n_pairs": int(len(scores)),
        "n_genuine": int(np.sum(labels == 1)),
        "n_impostor": int(np.sum(labels == 0)),
        "tar_at_far_0.001": float(tar_001),
        "tar_at_far_0.01": float(tar_01),
        "auc": float(auc),
        "eer": float(eer),
        "eer_threshold": float(eer_thr),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="LFW single-face subset eval (TAR@FAR)")
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    parser.add_argument("--pairs", default=str(repo / "lfw_data" / "pairs.csv"))
    parser.add_argument("--images-root", default=str(repo / "lfw_data" / "lfw"))
    parser.add_argument("--layout", choices=["raw", "marcelohaps"], default="raw")
    parser.add_argument("--det-size", type=int, default=320)
    parser.add_argument("--cache-dir", default=str(here / "cache"))
    parser.add_argument("--out", default=str(here / "out"))
    parser.add_argument("--report-name", default="lfw_single_face")
    parser.add_argument("--force-counts", action="store_true", help="перестроить кеш face-counts")
    args = parser.parse_args(argv)

    # 1) кеш эмбеддингов run_lfw.py
    model_path = Path(settings.MODELS_DIR) / settings.ARCFACE_MODEL_REL
    if not model_path.is_file():
        # Локально MODELS_DIR может быть Docker-путём (/app/models) и не резолвиться.
        # Кеш embeddings уже идентифицирован slug — sha нужен только для отчёта.
        logger.warning("ArcFace model not found at %s — sha will be 'unavailable'", model_path)
        slug = _model_slug(model_path)
        sha = "unavailable"
    else:
        slug = _model_slug(model_path)
        sha = _model_sha256(model_path)
    epath = cache_path(args.cache_dir, slug, args.det_size)
    if not epath.is_file():
        raise SystemExit(f"embeddings cache not found: {epath} (запусти run_lfw.py сначала)")
    with np.load(epath, allow_pickle=True) as d:
        emb_paths = list(d["paths"])
        emb_arr = np.asarray(d["embeddings"], dtype=np.float32)
    emb_map = {str(emb_paths[i]): emb_arr[i] for i in range(len(emb_paths))}
    logger.info("embeddings cache: %s (%d)", epath, len(emb_map))

    # 2) face-counts (детекция без encode)
    counts = compute_face_counts(
        list(emb_map.keys()), args.det_size, args.cache_dir, slug, args.force_counts)

    # 3) pairs → scores по срезам
    pairs = parse_pairs(Path(args.pairs), Path(args.images_root), layout=args.layout)
    logger.info("pairs: %d", len(pairs))

    all_s, all_l = [], []
    ss_s, ss_l = [], []      # single-single
    mi_s, mi_l = [], []      # multi-involved (хотя бы одно multi-face)
    n_missing = 0
    n_no_face_pair = 0
    for p in pairs:
        a, b = str(p["a"]), str(p["b"])
        ea, eb = emb_map.get(a), emb_map.get(b)
        if ea is None or eb is None:
            n_missing += 1
            continue
        ca, cb = counts.get(a, 0), counts.get(b, 0)
        if ca == 0 or cb == 0:
            n_no_face_pair += 1
            # no-face пара не в кеше эмбеддингов (уже исключена), но counts может дать 0
            continue
        score = float(np.dot(ea, eb))
        label = p["label"]
        all_s.append(score); all_l.append(label)
        if ca == 1 and cb == 1:
            ss_s.append(score); ss_l.append(label)
        elif ca > 1 or cb > 1:
            mi_s.append(score); mi_l.append(label)

    logger.info("scored: all=%d single-single=%d multi-involved=%d (missing=%d no_face_pair=%d)",
                len(all_s), len(ss_s), len(mi_s), n_missing, n_no_face_pair)

    all_s = np.asarray(all_s, dtype=np.float64); all_l = np.asarray(all_l, dtype=np.int64)
    ss_s = np.asarray(ss_s, dtype=np.float64); ss_l = np.asarray(ss_l, dtype=np.int64)
    mi_s = np.asarray(mi_s, dtype=np.float64); mi_l = np.asarray(mi_l, dtype=np.int64)

    res_all = _subset_metrics(all_s, all_l)
    res_ss = _subset_metrics(ss_s, ss_l)
    res_mi = _subset_metrics(mi_s, mi_l)

    # 4) сводка в stdout
    print("=" * 74)
    print(f"LFW single-face subset eval - model={slug} det={args.det_size}")
    print("-" * 74)
    for name, r in [("ALL", res_all), ("SINGLE-SINGLE", res_ss), ("MULTI-INVOLVED", res_mi)]:
        if "error" in r:
            print(f"  {name:16s}: {r['error']}")
            continue
        meet = "MEET" if r["tar_at_far_0.001"] >= 0.99 else "MISS"
        print(f"  {name:16s}: n={r['n_pairs']:5d} (g={r['n_genuine']}, i={r['n_impostor']})  "
              f"TAR@FAR=0.001={r['tar_at_far_0.001']:.4f} [{meet}]  AUC={r['auc']:.4f} EER={r['eer']:.4f}")
    print("-" * 74)
    print("  TZ target TAR>=0.99 @ FAR<=0.001:")
    print(f"    ALL           = {res_all.get('tar_at_far_0.001', 0):.4f} "
          f"[{'MEET' if res_all.get('tar_at_far_0.001',0)>=0.99 else 'MISS'}]")
    print(f"    SINGLE-SINGLE = {res_ss.get('tar_at_far_0.001', 0):.4f} "
          f"[{'MEET' if res_ss.get('tar_at_far_0.001',0)>=0.99 else 'MISS'}]")
    print("=" * 74)

    # 5) отчёт
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model_slug": slug,
        "model_sha256": sha,
        "det_size": args.det_size,
        "n_pairs_total": len(pairs),
        "n_missing": n_missing,
        "n_no_face_pair": n_no_face_pair,
        "all": res_all,
        "single_single": res_ss,
        "multi_involved": res_mi,
        "tz_target_tar_at_far_0_001": 0.99,
        "note": (
            "ALL = все пары; SINGLE-SINGLE = оба фото ровно 1 лицо; "
            "MULTI-INVOLVED = хотя бы одно фото multi-face. Memory lfw-single-face-meets-target "
            "утверждала SINGLE-SINGLE = 0.9964 (3927 пар). Этот замер подтверждает/опровергает."
        ),
    }
    rpath = out_dir / f"{args.report_name}_report.json"
    with open(rpath, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logger.info("report: %s", rpath)


if __name__ == "__main__":
    main()