# evaluation/liveness/report.py — отчёты liveness eval (детерминированный JSON + CSV).
#
# Переиспользует _to_jsonable/write_json/write_csv из evaluation.report.
# JSON: <name>_liveness_report.json — dataset_meta, thresholds, liveness-сводка,
#   per-type таблица, at_current(0.5)/at_recommended, note (smoke-статус).
# CSV: <name>_liveness_roc.csv (threshold,apcer,npcer,accuracy),
#      <name>_liveness_score_dist.csv (per-class гистограммы real_score).
# JSON без volatile-полей → повторный прогон из кеша байт-идентичен.

from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation.report import _to_jsonable, write_csv, write_json

OUT_DIR_DEFAULT = "evaluation/liveness/out"


def build_liveness_payload(dataset_meta: dict, result: dict, note: str | None = None) -> dict:
    """Сводный payload из eval_liveness. Сырые массивы (roc/score_dist/scores)
    НЕ кладутся в JSON — они уходят в CSV; остаются сводные числа + per-type."""
    payload: dict = {"dataset_meta": _to_jsonable(dataset_meta)}

    if "error" in result:
        payload["liveness"] = {"error": result["error"]}
        return payload

    rec = result["recommended"]
    cur = result["at_current"]
    rec_at = result["at_recommended"]

    payload["thresholds"] = {
        "current": cur["threshold"],
        "recommended": rec["threshold"],
        "eer_threshold": rec["eer_threshold"],
        "basis": rec["basis"],
    }

    payload["liveness"] = {
        "n_live": result["n_live"],
        "n_attack": result["n_attack"],
        "accuracy_at_current": cur["accuracy"],
        "accuracy_at_recommended": rec_at["accuracy"],
        "npcer_at_current": cur["npcer"],
        "npcer_at_recommended": rec_at["npcer"],
        "acer_overall_at_current": cur["acer"],
        "acer_overall_at_recommended": rec_at["acer"],
        "apcer_max_at_current": cur["apcer_max"],
        "apcer_max_at_recommended": rec_at["apcer_max"],
        "eer": rec["eer"],
        "auc": rec["auc"],
        "per_type": _per_type_table(cur, rec_at),
        "at_current": {
            "threshold": cur["threshold"],
            "apcer_per_type": cur["apcer_per_type"],
            "apcer_max": cur["apcer_max"],
            "npcer": cur["npcer"],
            "acer": cur["acer"],
            "accuracy": cur["accuracy"],
        },
        "at_recommended": {
            "threshold": rec_at["threshold"],
            "apcer_per_type": rec_at["apcer_per_type"],
            "apcer_max": rec_at["apcer_max"],
            "npcer": rec_at["npcer"],
            "acer": rec_at["acer"],
            "accuracy": rec_at["accuracy"],
        },
    }

    if note:
        payload["liveness"]["note"] = note

    # целевой KPI из ТЗ: liveness accuracy ≥ 98% — отдельный явный флаг.
    payload["kpi"] = {
        "target_accuracy": 0.98,
        "accuracy_at_current": cur["accuracy"],
        "accuracy_at_recommended": rec_at["accuracy"],
        "target_met_at_current": bool(cur["accuracy"] >= 0.98),
        "target_met_at_recommended": bool(rec_at["accuracy"] >= 0.98),
    }
    return payload


def _per_type_table(at_current: dict, at_recommended: dict) -> dict:
    out = {}
    for atype, v in at_current["apcer_per_type"].items():
        rv = at_recommended["apcer_per_type"].get(atype, {"n": 0, "apcer": 0.0})
        out[atype] = {
            "n": v["n"],
            "apcer_at_current": v["apcer"],
            "apcer_at_recommended": rv["apcer"],
        }
    return out


def write_liveness_csvs(out_dir: str | Path, name: str, result: dict) -> list[Path]:
    """Пишет roc + score_dist CSV. Возвращает список созданных путей."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if "error" in result:
        return written

    roc = result["roc"]
    apcer = np.asarray(roc["apcer"])
    tpr = np.asarray(roc["tpr"])
    thr = np.asarray(roc["thresholds"])
    npcer = 1.0 - tpr
    # ROC: (threshold, apcer, npcer, tpr). accuracy при пороге thr считается отдельно
    # (есть в JSON) — в CSV отдаём ROC-кривую, чего достаточно для построения графика.
    p = out_dir / f"{name}_roc.csv"
    write_csv(p, ["threshold", "apcer", "npcer", "tpr"], zip(thr, apcer, npcer, tpr))
    written.append(p)

    sd = result["score_dist"]
    bl = np.asarray(sd["bin_low"])
    bh = np.asarray(sd["bin_high"])
    live = np.asarray(sd["live"])
    attack = np.asarray(sd["attack"])
    p = out_dir / f"{name}_score_dist.csv"
    write_csv(p, ["bin_low", "bin_high", "live", "attack"], zip(bl, bh, live, attack))
    written.append(p)

    return written


def write_liveness_report(
    out_dir: str | Path,
    name: str,
    dataset_meta: dict,
    result: dict,
    note: str | None = None,
) -> dict:
    """Полная запись: JSON + CSV. Возвращает {json, csvs}."""
    out_dir = Path(out_dir)
    payload = build_liveness_payload(dataset_meta, result, note=note)
    json_path = out_dir / f"{name}_report.json"
    write_json(json_path, payload)
    csvs = write_liveness_csvs(out_dir, name, result)
    return {"json": json_path, "csvs": csvs}