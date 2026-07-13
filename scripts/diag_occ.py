# scripts/diag_occ.py — диагностика raw HSV-сигналов окклюзии по 3 классам.
# Цель: увидеть РЕАЛЬНЫЕ распределения skin_ref, lower-zone, eye/cheek band на
# чистых / очках / маске, чтобы выбрать метрики, которые разделяют (а не гадать).
#
# Использование:
#   python scripts/diag_occ.py <clean_dir> <sunglasses_dir> <mask_dir>
#     каждый dir — каталог фото; скрипт берёт основное лицо (крупнейшее),
#   original-кроп (как production после crop-from-original), считает сигналы.
#
# 152-ФЗ: только числовые сигналы, без кадров/эмбеддингов. UTF-8 stdout.
from __future__ import annotations

import os
import sys
from pathlib import Path

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

_EXPAND = 0.30


def _signals(files: list[Path], preprocessor, detector, gate) -> list[dict]:
    out: list[dict] = []
    for f in files:
        try:
            original = preprocessor.decode(f.read_bytes())
            downscaled = preprocessor.process_image(original)
        except Exception:
            continue
        faces = detector.detect(downscaled) or []
        if not faces:
            continue
        face = max(faces, key=lambda fc: (fc["bbox"][2]-fc["bbox"][0])*(fc["bbox"][3]-fc["bbox"][1]))
        if face["confidence"] < 0.75:
            continue
        scaled = scale_faces_to_original([face], original.shape, downscaled.shape)[0]
        x1, y1, x2, y2 = map(int, scaled["bbox"])
        x1, y1, x2, y2 = expand_bbox((x1, y1, x2, y2), _EXPAND, original.shape)
        crop, origin = gate._face_crop(original, (x1, y1, x2, y2))
        if crop is None or min(crop.shape[:2]) < gate.occ_min_face_side:
            continue
        pts = gate._landmarks_in_crop(scaled.get("landmarks"), origin, crop.shape[:2])
        if pts is None:
            continue
        le, re_, nose, ml, mr = pts[0], pts[1], pts[2], pts[3], pts[4]
        h, w = crop.shape[:2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        eye_dist = float(abs(re_[0]-le[0]))
        eye_y = float((le[1]+re_[1])/2.0)
        # Эталон: переносица (между глаз и носом)
        rxl = int(max(0, min(le[0], re_[0]) - 0.10*eye_dist))
        rxr = int(min(w, max(le[0], re_[0]) + 0.10*eye_dist))
        ry0 = int(max(0, min(eye_y + 0.15*eye_dist, h)))
        ry1 = int(min(h, max(ry0+1, nose[1] - 0.05*eye_dist)))
        ref = hsv[ry0:ry1, rxl:rxr] if ry1 > ry0 and rxr > rxl else None

        # Lower zone: нос→низ (ширина mouth)
        mouth_dx = float(abs(mr[0]-ml[0]))
        mouth_y = float((ml[1]+mr[1])/2.0)
        lx = int(max(0, min(ml[0], mr[0]) - 0.15*mouth_dx))
        rx = int(min(w, max(ml[0], mr[0]) + 0.15*mouth_dx))
        ly0 = int(max(0, min(nose[1], h)))
        ly1 = int(min(h, max(ly0+1, int(h*0.95))))
        chin0 = int(max(0, min(mouth_y + 0.30*max(1.0, mouth_y-nose[1]), h)))
        chin1 = int(min(h, max(chin0+1, int(h*0.97))))
        lower = hsv[ly0:ly1, lx:rx] if ly1 > ly0 and rx > lx else None
        chin = hsv[chin0:chin1, lx:rx] if chin1 > chin0 and rx > lx else None

        # Eye/cheek bands (как dark-eyes)
        half = max(2, int(eye_dist*0.22))
        xl = int(max(0, min(le[0], re_[0]) - 0.10*eye_dist))
        xr = int(min(w, max(le[0], re_[0]) + 0.10*eye_dist))
        ey0 = max(0, int(eye_y - half*0.6)); ey1 = min(h, int(eye_y + half*0.6))
        ch0 = max(0, int(eye_y + half*1.2)); ch1 = min(h, int(eye_y + half*2.0))
        eye_band = hsv[ey0:ey1, xl:xr] if ey1 > ey0 and xr > xl else None
        cheek_band = hsv[ch0:ch1, xl:xr] if ch1 > ch0 and xr > xl else None

        def m(region, ch):
            return float(region[:, :, ch].mean()) if region is not None and region.size else float("nan")

        d = {
            "name": f.name[-10:],
            "ref_H": m(ref, 0) if ref is not None and ref.size else float("nan"),
            "ref_S": m(ref, 1) if ref is not None and ref.size else float("nan"),
            "ref_Sz": int(ref.size) if ref is not None else 0,
            "low_H": m(lower, 0), "low_S": m(lower, 1), "low_V": m(lower, 2),
            "low_Sz": int(lower.size) if lower is not None else 0,
            "chin_H": m(chin, 0), "chin_S": m(chin, 1), "chin_V": m(chin, 2),
            "chin_Sz": int(chin.size) if chin is not None else 0,
            "eye_V": m(eye_band, 2), "eye_S": m(eye_band, 1),
            "che_V": m(cheek_band, 2), "che_S": m(cheek_band, 1),
            "eye_dark": (m(eye_band,2)/m(cheek_band,2)) if eye_band is not None and cheek_band is not None and m(cheek_band,2)>1e-3 else float("nan"),
            "sat_drop": (m(eye_band,1)/m(cheek_band,1)) if eye_band is not None and cheek_band is not None and m(cheek_band,1)>1e-3 else float("nan"),
        }
        # frac H-only (кожа vs эталон) и H+absSmin в lower-zone
        if lower is not None and ref is not None and ref.size:
            Href = float(np.median(ref[:, :, 0])); Sref = float(np.median(ref[:, :, 1]))
            Hc = lower[:, :, 0].astype(np.float32); Sc = lower[:, :, 1].astype(np.float32)
            dh = np.minimum(np.abs(Hc-Href), 180.0-np.abs(Hc-Href))
            d["frac_H"] = float((dh < 20).sum())/lower.shape[0]/lower.shape[1]
            d["frac_H_Sabs"] = float(((dh < 20) & (Sc > 15)).sum())/lower.shape[0]/lower.shape[1]
            d["frac_H_Srel"] = float(((dh < 20) & (np.abs(Sc-Sref) < 50)).sum())/lower.shape[0]/lower.shape[1]
        else:
            d["frac_H"] = d["frac_H_Sabs"] = d["frac_H_Srel"] = float("nan")
        # ref_V + relative V ratio (lower/ref) — маска затемняет нижнюю зону vs кожа-эталон
        ref_V = float(np.median(ref[:, :, 2])) if ref is not None and ref.size else float("nan")
        d["ref_V"] = ref_V
        low_V_mean = m(lower, 2) if lower is not None and lower.size else float("nan")
        d["v_ratio"] = (low_V_mean/ref_V) if (ref_V==ref_V and ref_V>1e-3 and low_V_mean==low_V_mean) else float("nan")
        out.append(d)
    return out


def _stats(label: str, rows: list[dict], key: str) -> str:
    vals = [r[key] for r in rows if r.get(key) == r.get(key)]  # filter NaN
    if not vals:
        return f"{label:>16}: НЕТ"
    return f"{label:>16}: n={len(vals)} min={min(vals):.3f} max={max(vals):.3f} med={float(np.median(vals)):.3f}"


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 3:
        print("usage: diag_occ.py <clean_dir> <sunglasses_dir> <mask_dir> [clean_range] [sun_range] [mask_range]")
        print("  ranges: 'a-b' (1-based inclusive), default all")
        return
    pre = ImagePreprocessor(); det = RetinaFaceDetector(det_size=settings.RETINA_DET_SIZE)
    gate = ImageQualityGate()

    def _range(spec: str) -> tuple[int, int] | None:
        if not spec:
            return None
        a, b = spec.split("-")
        return int(a), int(b)

    rng = {"clean": _range(args[3]) if len(args) > 3 else None,
           "sunglasses": _range(args[4]) if len(args) > 4 else None,
           "mask": _range(args[5]) if len(args) > 5 else None}
    groups = {"clean": args[0], "sunglasses": args[1], "mask": args[2]}
    data: dict[str, list[dict]] = {}
    for gname, gdir in groups.items():
        files = sorted([Path(p) for p in Path(gdir).iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}])
        r = rng[gname]
        if r is not None:
            files = files[r[0]-1:r[1]]
        data[gname] = _signals(files, pre, det, gate)
        print(f"\n=== {gname} ({gdir}) — {len(data[gname])} кадров ===")
        for r in data[gname]:
            print(f"  {r['name']} refH={r['ref_H']:.0f} refS={r['ref_S']:.0f} "
                  f"lowH={r['low_H']:.0f} lowS={r['low_S']:.0f} lowV={r['low_V']:.0f} "
                  f"chinH={r['chin_H']:.0f} chinS={r['chin_S']:.0f} chinSz={r['chin_Sz']} "
                  f"eyeV={r['eye_V']:.0f} eyeS={r['eye_S']:.0f} cheV={r['che_V']:.0f} cheS={r['che_S']:.0f} "
                  f"dark={r['eye_dark']:.3f} satd={r['sat_drop']:.3f} "
                  f"fracH={r['frac_H']:.3f} fracHS={r['frac_H_Sabs']:.3f}")

    print("\n=== СВОДКА (медианы диапазонов) ===")
    keys = ["eye_dark", "sat_drop", "frac_H", "frac_H_Sabs", "low_S", "low_V", "v_ratio", "ref_S"]
    for key in keys:
        for g in ["clean", "sunglasses", "mask"]:
            print(_stats(g, data[g], key))


if __name__ == "__main__":
    main()