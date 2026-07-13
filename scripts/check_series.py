# scripts/check_series.py — диагностический прогон полного pipeline на серии
# кадров (Camera Roll). Цель: понять, почему чистые лица (без маски/очков, нормальный
# свет) бракуются как "снимите маску" (remove_occlusion) или "плохой снимок"
# (quality_reject) или "spoof" (liveness).
#
# Использование:
#   python scripts/check_series.py [<dir>] [start_idx] [end_idx] [reencode_q]
#     dir        — каталог с фото (по умолчанию "Camera Roll")
#     start_idx  — 1-based индекс первого файла (по умолчанию 1)
#     end_idx    — 1-based индекс последнего файла (по умолчанию все)
#     reencode_q — если задано (>0), имитирует путь demo: cv2.imread ->
#                  cv2.imencode(".jpg", q) -> bytes -> process. По умолчанию 0 (читать файл как есть).
#
# Контракт 152-ФЗ: читает фото только локально, выводит только числовые сигналы
# (eye_dark, skin_frac, live_score, blur, face_wh) — без кадров/base64/эмбеддингов.
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MODELS_DIR", str(ROOT / "models"))
os.environ.setdefault("LIVENESS_ENABLED", "true")
os.environ.setdefault("AUTH_ENABLED", "false")

from app.ml.pipeline_v2 import FacePipelineV2  # noqa: E402


def main() -> None:
    args = sys.argv[1:]
    src_dir = Path(args[0]) if len(args) > 0 else (ROOT / "Camera Roll")
    start = int(args[1]) if len(args) > 1 else 1
    end = int(args[2]) if len(args) > 2 else 10**9
    reencode_q = int(args[3]) if len(args) > 3 else 0
    if reencode_q > 0:
        print(f"[режим] имитация demo-пути: cv2.imread -> imencode(.jpg, q={reencode_q}) -> bytes -> process")

    files = sorted(
        [f for f in src_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    sel = files[start - 1 : end]
    if not sel:
        print(f"Нет файлов в {src_dir} [{start}:{end}]")
        return

    pipe = FacePipelineV2()
    pipe._init()

    # Шапка: idx | status | reason | eye_dark | vratio | sun | mask | edge | bright | live | face_wh
    print(
        f"{'idx':>3} {'status':<14} {'reason':<18} "
        f"{'eye_dark':>8} {'vratio':>6} {'sun':>4} {'mask':>5} "
        f"{'glasses':>7} {'edge':>6} {'bright':>6} "
        f"{'live':>6} {'face_wh':>10}"
    )
    print("-" * 116)

    counts: dict[str, int] = {}
    for i, f in enumerate(sel, start=start):
        try:
            if reencode_q > 0:
                frame = cv2.imread(str(f))
                if frame is None:
                    raise RuntimeError("imread failed")
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, reencode_q])
                if not ok:
                    raise RuntimeError("imencode failed")
                payload = buf.tobytes()
            else:
                payload = f.read_bytes()
            res = pipe.process(payload)
        except Exception as e:
            status, reason = "error", str(e)[:40]
            print(f"{i:>3} {status:<14} {reason:<18}")
            counts["error"] = counts.get("error", 0) + 1
            continue

        status = res.get("status", "?")
        counts[status] = counts.get(status, 0) + 1
        qd = res.get("quality_details", {}) or {}
        occ = qd.get("occlusion_flags", {}) or {}
        reason = res.get("quality_reason") or res.get("liveness_reason") or "-"
        eye_dark = occ.get("eye_dark_ratio")
        vratio = occ.get("lower_face_v_ratio")
        edge = occ.get("eye_edge_density")
        sun = occ.get("sunglasses_detected")
        glasses = occ.get("glasses_detected")
        mask = occ.get("mask_detected")
        bright = qd.get("brightness")
        live = res.get("liveness_score")
        bbox = res.get("bbox") or [0, 0, 0, 0]
        fw = int(bbox[2] - bbox[0]) if bbox else 0
        fh = int(bbox[3] - bbox[1]) if bbox else 0

        def fmt(v, w, p=3):
            if v is None:
                return f"{'-':>{w}}"
            return f"{float(v):>{w}.{p}f}"

        print(
            f"{i:>3} {status:<14} {reason:<18} "
            f"{fmt(eye_dark,8)} {fmt(vratio,6)} {str(sun):>4} {str(mask):>5} "
            f"{str(glasses):>7} {fmt(edge,6)} {fmt(bright,6,1)} "
            f"{fmt(live,6)} {f'{fw}x{fh}':>10}"
        )

    print("-" * 116)
    print("Итоги по статусам:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()