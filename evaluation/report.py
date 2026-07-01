# evaluation/report.py — запись отчётов eval-harness (детерминированный JSON + CSV).
#
# JSON: <name>_report.json — сводная схема (dataset_meta, thresholds, 1:1, 1:N).
#   Детерминированный (sort_keys, без volatile-timestamps) → повторный прогон
#   даёт байт-идентичный файл (см. план P1, шаг 9 верификации).
# CSV: <name>_roc.csv (threshold,far,tar,frr),
#      <name>_cmc.csv (rank,accuracy),
#      <name>_score_dist.csv (bin_low,bin_high,genuine,impostor). Формат %.6g.

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

OUT_DIR_DEFAULT = "evaluation/out"


# ---------------- JSON-сериализация -------------------------------------------

def _to_jsonable(obj):
    """Рекурсивно превращает numpy-типы в нативные (для json.dump)."""
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


def write_json(path: str | os.PathLike, payload: dict) -> None:
    """Детерминированная запись JSON (sort_keys, indent=2, ensure_ascii=False)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_to_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")


# ---------------- CSV ---------------------------------------------------------

def _fmt(value) -> str:
    """Формат %.6g для чисел; иначе str."""
    if isinstance(value, (int, float)) or isinstance(value, (np.floating, np.integer)):
        return f"{float(value):.6g}"
    return str(value)


def write_csv(path: str | os.PathLike, header: list[str], rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([_fmt(v) for v in row])


# ---------------- сборка отчёта ----------------------------------------------

def build_report_payload(
    dataset_meta: dict,
    one_to_one: dict | None,
    one_to_n: dict | None,
    faiss_check: dict | None = None,
) -> dict:
    """
    Собирает сводный payload в схеме плана из результатов eval_1to1 / eval_1toN.
    Сырые массивы (roc/cmc/score_dist/scores/labels) намеренно НЕ включаются в
    JSON — они уходят в CSV; в JSON остаются только сводные числа.
    """
    payload: dict = {"dataset_meta": _to_jsonable(dataset_meta)}

    if one_to_one and "error" not in one_to_one:
        rec = one_to_one["thresholds"]["recommended"]
        cur = one_to_one["thresholds"]["current"]
        payload["thresholds"] = {
            "current": cur,
            "recommended": rec,
            "basis": rec["basis"],
        }
        payload["verification_1to1"] = {
            "n_genuine": one_to_one["n_genuine"],
            "n_impostor": one_to_one["n_impostor"],
            "impostor_ratio": one_to_one["impostor_ratio"],
            "seed": one_to_one["seed"],
            "tar_at_far_0_001": one_to_one["tar_at_far"],
            "frr_at_recommended_high": one_to_one["frr_at_recommended_high"],
            "far_at_recommended_high": one_to_one["far_at_recommended_high"],
            "eer": one_to_one["eer"],
            "auc": one_to_one["auc"],
            "at_current_high": one_to_one["at_current_high"],
        }
    elif one_to_one:
        payload["verification_1to1"] = {"error": one_to_one.get("error")}

    if one_to_n and "error" not in one_to_n:
        cmc_acc = np.asarray(one_to_n["cmc"]["accuracy"])
        ranks = np.asarray(one_to_n["cmc"]["ranks"])
        top5 = [
            {"rank": int(r), "accuracy": float(a)}
            for r, a in list(zip(ranks, cmc_acc))[:5]
        ]
        ident = {
            "n_gallery": one_to_n["n_gallery"],
            "n_probes": one_to_n["n_probes"],
            "n_ids_total": one_to_n["n_ids_total"],
            "n_ids_single_image": one_to_n["n_ids_single_image"],
            "rank1": one_to_n["rank1"],
            "rank5": one_to_n["rank5"],
            "cmc_top5": top5,
        }
        if faiss_check and "error" not in faiss_check:
            ident["faiss_vs_numpy_mismatch"] = faiss_check["mismatch"]
            ident["faiss_check"] = faiss_check
        payload["identification_1toN"] = ident
    elif one_to_n:
        payload["identification_1toN"] = {"error": one_to_n.get("error")}

    return payload


def write_csvs(out_dir: str | os.PathLike, name: str, one_to_one: dict | None, one_to_n: dict | None) -> list[Path]:
    """Пишет roc/cmc/score_dist CSV. Возвращает список созданных путей."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if one_to_one and "error" not in one_to_one:
        roc = one_to_one["roc"]
        far = np.asarray(roc["far"])
        tar = np.asarray(roc["tar"])
        frr = np.asarray(roc["frr"])
        thr = np.asarray(roc["thresholds"])
        rows = zip(thr, far, tar, frr)
        p = out_dir / f"{name}_roc.csv"
        write_csv(p, ["threshold", "far", "tar", "frr"], rows)
        written.append(p)

        sd = one_to_one["score_dist"]
        bl = np.asarray(sd["bin_low"])
        bh = np.asarray(sd["bin_high"])
        g = np.asarray(sd["genuine"])
        imp = np.asarray(sd["impostor"])
        p = out_dir / f"{name}_score_dist.csv"
        write_csv(p, ["bin_low", "bin_high", "genuine", "impostor"], zip(bl, bh, g, imp))
        written.append(p)

    if one_to_n and "error" not in one_to_n:
        ranks = np.asarray(one_to_n["cmc"]["ranks"])
        acc = np.asarray(one_to_n["cmc"]["accuracy"])
        p = out_dir / f"{name}_cmc.csv"
        write_csv(p, ["rank", "accuracy"], zip(ranks, acc))
        written.append(p)

    return written


def write_report(
    out_dir: str | os.PathLike,
    name: str,
    dataset_meta: dict,
    one_to_one: dict | None = None,
    one_to_n: dict | None = None,
    faiss_check: dict | None = None,
) -> dict:
    """
    Полная запись отчёта: JSON + CSV. Возвращает {json, csvs}.
    """
    out_dir = Path(out_dir)
    payload = build_report_payload(dataset_meta, one_to_one, one_to_n, faiss_check)
    json_path = out_dir / f"{name}_report.json"
    write_json(json_path, payload)
    csvs = write_csvs(out_dir, name, one_to_one, one_to_n)
    return {"json": json_path, "csvs": csvs}