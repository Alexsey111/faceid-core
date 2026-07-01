# evaluation/liveness/run_eval.py — CLI: прогон liveness по anti-spoofing-датасету.
#
# Поток: build_liveness_samples → (cache hit? иначе) детекция+predict по всем кадрам
# → кеш .npz (scores/labels/attack_types/face_detected/meta) → eval_liveness → отчёт.
# no-face кадры skip'аются (нельзя скорить liveness без лица), считаются per class.
# Повторный прогон из кеша → байт-идентичный JSON (ONNX-CPU детерминирован).

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ensure_models_env должен вызываться до импорта app.* (читает MODELS_DIR env).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # git-root (содержит app/) on path
from evaluation.cli_common import REPO_ROOT, ensure_models_env  # noqa: E402

ensure_models_env()

from app.core.config import settings  # noqa: E402
from app.ml.detection.retinaface_detector import RetinaFaceDetector  # noqa: E402
from app.ml.preprocessing.image_preprocessor import ImagePreprocessor  # noqa: E402

from evaluation.liveness.checkers import MODEL_REGISTRY, build_checker, resolve_model_path  # noqa: E402
from evaluation.liveness.dataset import build_liveness_samples, dataset_stats  # noqa: E402
from evaluation.liveness.predict import score_frame  # noqa: E402
from evaluation.liveness.protocols import eval_liveness  # noqa: E402
from evaluation.liveness.report import OUT_DIR_DEFAULT, write_liveness_report  # noqa: E402
from evaluation.liveness.video import iter_video_frames  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("liveness.run_eval")

CACHE_DIR_DEFAULT = "evaluation/liveness/cache"


# ---------------- кеш предсказаний --------------------------------------------

def _model_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path(cache_dir: str, name: str, n_frames: int, det_size: int, model_slug: str) -> Path:
    return Path(cache_dir) / f"{name}_nfr{n_frames}_det{det_size}_{model_slug}.npz"


def save_predictions(
    cache_dir: str, name: str, n_frames: int, det_size: int, model_slug: str,
    scores, labels, attack_types, face_detected, sources, meta: dict,
) -> Path:
    p = _cache_path(cache_dir, name, n_frames, det_size, model_slug)
    p.parent.mkdir(parents=True, exist_ok=True)
    import json
    np.savez(
        p,
        scores=np.asarray(scores, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        attack_types=np.asarray(attack_types, dtype=object),
        face_detected=np.asarray(face_detected, dtype=bool),
        sources=np.asarray(sources, dtype=object),
        meta_json=np.array(json.dumps(meta, sort_keys=True, ensure_ascii=False), dtype=object),
    )
    return p


def load_predictions(cache_dir: str, name: str, n_frames: int, det_size: int, model_slug: str):
    p = _cache_path(cache_dir, name, n_frames, det_size, model_slug)
    if not p.exists():
        return None
    import json
    with np.load(p, allow_pickle=True) as d:
        meta = json.loads(str(d["meta_json"].item()))
        return {
            "scores": np.array(d["scores"], dtype=np.float64),
            "labels": np.array(d["labels"], dtype=np.int64),
            "attack_types": np.array(d["attack_types"], dtype=object),
            "face_detected": np.array(d["face_detected"], dtype=bool),
            "sources": np.array(d["sources"], dtype=object),
            "meta": meta,
        }


# ---------------- прогон предсказаний -----------------------------------------

def _read_image_bgr(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img if img is not None else None


def run_predictions(
    samples, det_size: int, n_frames: int, model_slug: str,
    max_side: int = 480, progress_every: int = 50,
) -> dict:
    """
    Прогон детекции + liveness по всем сэмплам. Группирует видео-сэмплы по пути,
    читает кадры одного видео один раз. Возвращает arrays scores/labels/types/
    face_detected/sources + meta (n_no_face_per_class, ...).
    """
    det = RetinaFaceDetector(det_size=det_size)
    checker = build_checker(model_slug)
    model_path = resolve_model_path(model_slug)
    preproc = ImagePreprocessor()

    liveness_sha = _model_sha256(model_path)

    scores: list[float] = []
    labels: list[int] = []
    attack_types: list[str] = []
    face_detected: list[bool] = []
    sources: list[str] = []
    n_no_face_per_class: dict[str, int] = defaultdict(int)
    n_total_per_class: dict[str, int] = defaultdict(int)
    n_video_open_fail = 0

    # Группировка видео-сэмплов по пути (читать кадры один раз на видео).
    image_samples = [s for s in samples if s.is_image]
    video_groups: dict[Path, list] = defaultdict(list)
    for s in samples:
        if not s.is_image:
            video_groups[Path(s.path)].append(s)

    processed = 0

    # 1) изображения
    for s in image_samples:
        n_total_per_class[s.cls] += 1
        img = _read_image_bgr(Path(s.path))
        if img is None:
            n_no_face_per_class[s.cls] += 1
            processed += 1
            continue
        sc, fd = score_frame(img, det, checker, preproc)
        _record(sc, fd, s, scores, labels, attack_types, face_detected, sources, n_no_face_per_class)
        processed += 1
        if processed % progress_every == 0:
            logger.info("processed %d/%d (images)", processed, len(samples))

    # 2) видео — по группам
    for vpath, vsamples in video_groups.items():
        frames = list(iter_video_frames(vpath, n_frames, max_side=max_side))
        if not frames:
            n_video_open_fail += 1
            # все сэмплы этого видео → no-face
            for s in vsamples:
                n_total_per_class[s.cls] += 1
                n_no_face_per_class[s.cls] += 1
                processed += 1
            continue
        # vsamples упорядочены по frame_index 0..N-1 — соответствует порядку frames
        for i, s in enumerate(vsamples):
            n_total_per_class[s.cls] += 1
            if i >= len(frames):
                n_no_face_per_class[s.cls] += 1
                processed += 1
                continue
            sc, fd = score_frame(frames[i], det, checker, preproc)
            _record(sc, fd, s, scores, labels, attack_types, face_detected, sources, n_no_face_per_class)
            processed += 1
            if processed % progress_every == 0:
                logger.info("processed %d/%d (videos)", processed, len(samples))

    meta = {
        "n_samples": len(samples),
        "n_frames_per_video": n_frames,
        "det_size": det_size,
        "max_side": max_side,
        "model_slug": model_slug,
        "model_desc": MODEL_REGISTRY[model_slug]["desc"],
        "crop_scale": MODEL_REGISTRY[model_slug]["crop_scale"],
        "liveness_model_sha256": liveness_sha,
        "liveness_model_path": str(model_path),
        "n_no_face_per_class": dict(n_no_face_per_class),
        "n_total_per_class": dict(n_total_per_class),
        "n_video_open_fail": n_video_open_fail,
        "n_scored": len(scores),
        "n_no_face_total": sum(1 for x in face_detected if not x),
    }
    return {
        "scores": np.asarray(scores, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int64),
        "attack_types": np.asarray(attack_types, dtype=object),
        "face_detected": np.asarray(face_detected, dtype=bool),
        "sources": np.asarray(sources, dtype=object),
        "meta": meta,
    }


def _record(sc, fd, s, scores, labels, attack_types, face_detected, sources, n_no_face_per_class):
    if not fd or sc is None:
        # no-face → не попадает в метрики, но считаем per class
        n_no_face_per_class[s.cls] += 1
        return
    scores.append(float(sc))
    labels.append(int(s.label))
    attack_types.append(s.attack_type)
    face_detected.append(True)
    sources.append(str(s.path))


# ---------------- CLI ----------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Liveness eval-harness (anti-spoofing)")
    parser.add_argument("--root", default=str(REPO_ROOT / "Anti-Spoofing Dataset"))
    parser.add_argument("--n-frames", type=int, default=30)
    parser.add_argument("--det-size", type=int, default=320)
    parser.add_argument("--max-side", type=int, default=480)
    parser.add_argument("--model", default="yakhyo_v2", choices=list(MODEL_REGISTRY),
                        help="какая liveness-модель (контракт preprocess/crop зашит в чекер)")
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--out", default=OUT_DIR_DEFAULT)
    parser.add_argument("--report-name", default="liveness")
    parser.add_argument("--force", action="store_true", help="пересобрать кеш предсказаний")
    parser.add_argument("--smoke", action="store_true", help="малый прогон (по 1 файлу класса)")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"dataset root not found: {root}")

    samples = build_liveness_samples(root, n_frames_per_video=args.n_frames)
    if args.smoke:
        samples = _smoke_subset(samples)
    stats = dataset_stats(samples)
    logger.info("dataset: %s", stats)

    cached = None if args.force else load_predictions(
        args.cache_dir, args.report_name, args.n_frames, args.det_size, args.model)
    if cached is not None:
        logger.info("cache hit: %s", _cache_path(args.cache_dir, args.report_name, args.n_frames, args.det_size, args.model))
        pred = cached
    else:
        t0 = time.time()
        pred = run_predictions(samples, det_size=args.det_size, n_frames=args.n_frames,
                               model_slug=args.model, max_side=args.max_side)
        pred["meta"]["elapsed_s"] = round(time.time() - t0, 3)
        cpath = save_predictions(
            args.cache_dir, args.report_name, args.n_frames, args.det_size, args.model,
            pred["scores"], pred["labels"], pred["attack_types"],
            pred["face_detected"], pred["sources"], pred["meta"],
        )
        logger.info("cache saved: %s (%d scored frames, %.1fs)",
                    cpath, len(pred["scores"]), pred["meta"]["elapsed_s"])

    # метрики только на кадрах с обнаруженным лицом
    if len(pred["scores"]) == 0:
        logger.error("no scored frames (all no-face?) — nothing to evaluate")
        return
    result = eval_liveness(pred["scores"], pred["labels"], pred["attack_types"])

    n_sessions = _count_sessions(root)
    dataset_meta = {
        "root": str(root),
        "n_sessions": n_sessions,
        "n_frames_per_video": args.n_frames,
        "det_size": args.det_size,
        "max_side": args.max_side,
        "n_samples": stats["n_total"],
        "n_live": stats["n_live"],
        "n_attack": stats["n_attack"],
        "by_class": stats["by_class"],
        "by_attack_type": stats["by_attack_type"],
        "n_no_face_per_class": pred["meta"].get("n_no_face_per_class", {}),
        "n_total_per_class": pred["meta"].get("n_total_per_class", {}),
        "n_video_open_fail": pred["meta"].get("n_video_open_fail", 0),
        "n_scored_frames": int(len(pred["scores"])),
        "model_slug": pred["meta"].get("model_slug", args.model),
        "model_desc": pred["meta"].get("model_desc", ""),
        "crop_scale": pred["meta"].get("crop_scale", ""),
        "liveness_model_sha256": pred["meta"].get("liveness_model_sha256", ""),
    }
    note = ("smoke, %d sessions — метрики ознакомительные, доверительные интервалы широкие"
            % n_sessions) if n_sessions <= 12 else None

    out = write_liveness_report(args.out, args.report_name, dataset_meta, result, note=note)
    logger.info("report: %s", out["json"])
    _print_summary(result, dataset_meta)


def _smoke_subset(samples):
    """По одному файлу на класс (для --smoke)."""
    seen = set()
    out = []
    for s in samples:
        key = (s.cls, s.path)
        if s.path not in seen:
            seen.add(s.path)
            out.append(s)
    return out


def _count_sessions(root: Path) -> int:
    """Кол-во сессий съёмки = минимум файлов по распознанным классам
    (каждая сессия даёт по одному файлу в каждый класс: selfie/video/print/cutout/replay)."""
    from evaluation.liveness.dataset import CLASS_MAP
    counts = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        canon = None
        from evaluation.liveness.dataset import _resolve_class_dir_name
        canon = _resolve_class_dir_name(d.name)
        if canon is None:
            continue
        n = sum(1 for f in d.iterdir() if f.is_file())
        counts.append(n)
    return min(counts) if counts else 0


def _print_summary(result, dataset_meta):
    if "error" in result:
        print(f"[liveness] ERROR: {result['error']}")
        return
    cur = result["at_current"]
    rec = result["at_recommended"]
    r = result["recommended"]
    print("=" * 70)
    print(f"Liveness eval - {dataset_meta['n_sessions']} sessions, "
          f"{dataset_meta['n_scored_frames']} scored frames")
    print("-" * 70)
    print(f"  at current thr=0.5:  accuracy={cur['accuracy']:.4f}  ACER={cur['acer']:.4f}  "
          f"NPCER={cur['npcer']:.4f}  APCER_max={cur['apcer_max']:.4f}")
    print(f"  at recommended thr={r['threshold']:.4f}: accuracy={rec['accuracy']:.4f}  "
          f"ACER={rec['acer']:.4f}  NPCER={rec['npcer']:.4f}  APCER_max={rec['apcer_max']:.4f}")
    print(f"  EER={r['eer']:.4f}  AUC={r['auc']:.4f}  eer_threshold={r['eer_threshold']:.4f}")
    print("  per-type APCER @ current 0.5:")
    for atype, v in cur["apcer_per_type"].items():
        print(f"    {atype:8s} n={v['n']:4d}  APCER={v['apcer']:.4f}")
    noface = dataset_meta.get("n_no_face_per_class", {})
    if noface:
        total = dataset_meta.get("n_total_per_class", {})
        print("  no-face rate per class:")
        for cls, n in sorted(noface.items()):
            t = total.get(cls, 0)
            print(f"    {cls:22s} {n}/{t} ({n/t*100:.1f}%)" if t else f"    {cls}: {n}")
    target_acc = 0.98
    print("-" * 70)
    cur_meet = "MEET" if cur['accuracy'] >= target_acc else "MISS"
    rec_meet = "MEET" if rec['accuracy'] >= target_acc else "MISS"
    print(f"  TZ target accuracy>= {target_acc:.0%}: "
          f"current={cur_meet} ({cur['accuracy']:.2%})   "
          f"recommended={rec_meet} ({rec['accuracy']:.2%})")
    print("=" * 70)


if __name__ == "__main__":
    main()