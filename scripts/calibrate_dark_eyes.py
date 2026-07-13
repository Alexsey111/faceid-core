# scripts/calibrate_dark_eyes.py — калибровка QUALITY_MAX_EYE_DARK_RATIO на
# серии очки/без (original-кроп, как в production после переключения кропа из
# full-res). Старая калибровка 0.77 была на c110 (downscaled) и не обобщается:
# на live_selfie 5/8 чистых лиц < 0.77 → массовый False Retry.
#
# Использование:
#   python scripts/calibrate_dark_eyes.py <clean_dir> <sunglasses_dir>
#     clean_dir      — фото БЕЗ солнцезащитных очков (20–30 лиц, одна камера/свет)
#     sunglasses_dir — фото В солнцезащитных очках (20–30 лиц, та же камера/свет)
#
# Вывод: распределения eye_dark для обоих классов, перекрытие, ROC-точка
# разделения (max gap / min FRR@FAR) и рекомендуемый порог для settings.
#
# Контракт 152-ФЗ: скрипт читает фото только локально, ничего не логирует и не
# сохраняет биометрию — только числовой сигнал eye_dark_ratio.
from __future__ import annotations

import os
import sys
from pathlib import Path

# Windows console по умолчанию cp1251 — ломает кириллицу/юникод в выводе.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MODELS_DIR", str(ROOT / "models"))
os.environ.setdefault("LIVENESS_ENABLED", "false")
os.environ.setdefault("AUTH_ENABLED", "false")

from app.core.config import settings  # noqa: E402
from app.ml.detection.retinaface_detector import RetinaFaceDetector  # noqa: E402
from app.ml.pipeline_v2 import expand_bbox, scale_faces_to_original  # noqa: E402
from app.ml.preprocessing.image_preprocessor import ImagePreprocessor  # noqa: E402
from app.ml.quality.image_quality_gate import ImageQualityGate  # noqa: E402


_EXPAND = 0.30  # FAST_BBOX_EXPAND_SCALE — как в pipeline._prepare_face_from_detection


def _collect_eye_dark(
    files: list[Path],
    preprocessor: ImagePreprocessor,
    detector: RetinaFaceDetector,
    gate: ImageQualityGate,
) -> list[float]:
    """Прогнать фото через детекцию (downscaled) → scale к original → occ
    (dark-eyes) на original-кропе. Вернуть список eye_dark_ratio (только для
    детектированных чистых/крупных лиц)."""
    ratios: list[float] = []
    skipped = 0
    for f in files:
        try:
            original = preprocessor.decode(f.read_bytes())
            downscaled = preprocessor.process_image(original)
        except Exception:
            skipped += 1
            continue

        faces = detector.detect(downscaled) or []
        if not faces:
            skipped += 1
            continue
        # Берём основное лицо (как select_main_face в production — крупнейшее/自信).
        face = max(faces, key=lambda fc: (fc["bbox"][2] - fc["bbox"][0]) * (fc["bbox"][3] - fc["bbox"][1]))
        if face["confidence"] < 0.75:
            skipped += 1
            continue

        scaled = scale_faces_to_original([face], original.shape, downscaled.shape)[0]
        x1, y1, x2, y2 = map(int, scaled["bbox"])
        x1, y1, x2, y2 = expand_bbox((x1, y1, x2, y2), _EXPAND, original.shape)
        face_crop, origin = gate._face_crop(original, (x1, y1, x2, y2))
        if face_crop is None or face_crop.size == 0:
            skipped += 1
            continue
        if min(face_crop.shape[:2]) < gate.occ_min_face_side:
            skipped += 1  # мелкое лицо — dark-eyes пропускается в production
            continue
        pts = gate._landmarks_in_crop(
            scaled.get("landmarks"), origin, face_crop.shape[:2]
        )
        occ = gate._check_occlusion(face_crop, pts)
        r = occ.get("eye_dark_ratio")
        if r is not None:
            ratios.append(float(r))
        else:
            skipped += 1
    if skipped:
        print(f"  [skip] {skipped} фото без валидного лица/крупного кропа")
    return ratios


def _describe(label: str, rs: list[float]) -> None:
    if not rs:
        print(f"{label}: НЕТ данных")
        return
    print(
        f"{label}: n={len(rs)} min={min(rs):.3f} max={max(rs):.3f} "
        f"mean={np.mean(rs):.3f} median={np.median(rs):.3f}"
    )
    print(f"  sorted: {['%.3f' % r for r in sorted(rs)]}")


def _recommend_threshold(clean: list[float], glasses: list[float]) -> None:
    """Порог sunglasses_detected = eye_dark < threshold.
    Цель: чистые → pass (eye_dark >= thr), очки → retry (eye_dark < thr).
    Ищем порог, максимизирующий разделение (min FRR_clean при FAR_glasses=0,
    fallback — центр зазора между max(glasses) и min(clean))."""
    if not clean or not glasses:
        print("Недостаточно данных для рекомендации порога.")
        return

    candidates = sorted(set(clean + glasses))
    best = None  # (frr, far, thr)
    for thr in candidates:
        frr = np.mean([r < thr for r in clean])   # чистое ложно бракуем
        far = np.mean([r >= thr for r in glasses])  # очки ложно пропускаем
        # приоритет: FAR=0 (ни одни очки не пропущены), затем min FRR
        key = (far, frr)
        if best is None or key < best[0]:
            best = (key, thr, frr, far)

    _, thr, frr, far = best
    gap = min(clean) - max(glasses)
    print()
    print("=== Рекомендация ===")
    print(f"Зазор min(clean)={min(clean):.3f} - max(glasses)={max(glasses):.3f} = {gap:+.3f}")
    if gap <= 0:
        print("ВНИМАНИЕ: диапазоны ПЕРЕКРЫВАЮТСЯ — порог по eye/cheek ratio не разделяет.")
        print("Метрика eye_dark_ratio на этом датасете недостаточна. Рассмотрите")
        print("перепроектирование dark-eyes (HSV saturation по eye-band / детекция зрачка).")
    print(
        f"Лучший порог (FAR_очков=0, min FRR_чистых): threshold={thr:.3f}  "
        f"-> FRR_clean={frr:.1%} (чистых ложно бракуем), FAR_glasses={far:.1%} (очков пропускаем)"
    )
    print(f"Установить: QUALITY_MAX_EYE_DARK_RATIO={thr:.2f} в settings (.env).")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    clean_dir, glasses_dir = Path(sys.argv[1]), Path(sys.argv[2])
    clean_files = sorted(
        p for p in clean_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    glasses_files = sorted(
        p for p in glasses_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"clean: {len(clean_files)} фото из {clean_dir}")
    print(f"sunglasses: {len(glasses_files)} фото из {glasses_dir}")
    if not clean_files or not glasses_files:
        print("Один из каталогов пуст.")
        return 2

    preprocessor = ImagePreprocessor()
    detector = RetinaFaceDetector(det_size=settings.RETINA_DET_SIZE)
    gate = ImageQualityGate()

    print("\n--- clean (без очков) ---")
    clean = _collect_eye_dark(clean_files, preprocessor, detector, gate)
    _describe("clean", clean)

    print("\n--- sunglasses (в очках) ---")
    glasses = _collect_eye_dark(glasses_files, preprocessor, detector, gate)
    _describe("sunglasses", glasses)

    _recommend_threshold(clean, glasses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())