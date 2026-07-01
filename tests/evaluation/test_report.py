# tests/evaluation/test_report.py — unit-тесты report.py (JSON детерминизм + CSV).

import json

import numpy as np
import pytest

from evaluation.report import build_report_payload, write_csv, write_json, write_report

pytestmark = pytest.mark.unit


def _toy_1to1():
    return {
        "n_genuine": 5, "n_impostor": 50, "impostor_ratio": 10, "seed": 42,
        "target_far": 0.001,
        "tar_at_far": 0.99, "frr_at_recommended_high": 0.01,
        "far_at_recommended_high": 0.001, "eer": 0.02, "auc": 0.995,
        "thresholds": {
            "recommended": {"high": 0.55, "low": 0.40, "margin": 0.15,
                            "eer": 0.50, "basis": "test basis"},
            "current": {"high": 0.60, "low": 0.30, "margin": 0.10},
        },
        "at_current_high": {"far": 0.002, "frr": 0.02, "tar": 0.98, "threshold": 0.60},
        "roc": {"far": np.array([0.0, 0.5, 1.0]), "tar": np.array([0.0, 0.9, 1.0]),
                "frr": np.array([1.0, 0.1, 0.0]), "thresholds": np.array([2.0, 0.5, -1.0])},
        "score_dist": {"bin_low": np.array([-1.0, 0.0]), "bin_high": np.array([0.0, 1.0]),
                       "genuine": np.array([0, 5]), "impostor": np.array([50, 0])},
        "scores": np.array([0.9, 0.1]), "labels": np.array([1, 0]),
    }


def _toy_1toN():
    return {
        "n_gallery": 3, "n_probes": 4, "n_ids_total": 3, "n_ids_single_image": 1,
        "max_rank": 5, "rank1": 1.0, "rank5": 1.0,
        "cmc": {"ranks": np.array([1, 2, 3]), "accuracy": np.array([0.75, 1.0, 1.0])},
    }


def test_write_json_deterministic(tmp_path):
    p = tmp_path / "r.json"
    write_json(p, {"b": np.array([1, 2]), "a": 0.5, "c": np.float32(1.0)})
    t1 = p.read_text(encoding="utf-8")
    write_json(p, {"a": 0.5, "b": np.array([1, 2]), "c": np.float32(1.0)})  # другой порядок ключей
    t2 = p.read_text(encoding="utf-8")
    assert t1 == t2  # sort_keys → порядок не важен
    parsed = json.loads(t1)
    assert parsed["a"] == 0.5 and parsed["b"] == [1, 2] and parsed["c"] == 1.0


def test_write_csv_format(tmp_path):
    p = tmp_path / "x.csv"
    write_csv(p, ["threshold", "far"], [(0.123456789, 0.5), (1, 2)])
    txt = p.read_text(encoding="utf-8")
    lines = txt.strip().splitlines()
    assert lines[0] == "threshold,far"
    assert lines[1].startswith("0.123457")  # %.6g
    assert lines[2] == "1,2"


def test_build_report_payload_structure():
    p = build_report_payload({"n_ids": 10}, _toy_1to1(), _toy_1toN(), {"mismatch": 0, "n_probes": 4})
    assert p["dataset_meta"]["n_ids"] == 10
    assert "verification_1to1" in p and "identification_1toN" in p
    assert p["verification_1to1"]["tar_at_far_0_001"] == 0.99
    assert p["identification_1toN"]["rank1"] == 1.0
    assert p["identification_1toN"]["faiss_vs_numpy_mismatch"] == 0
    assert len(p["identification_1toN"]["cmc_top5"]) == 3


def test_write_report_creates_files(tmp_path):
    res = write_report(tmp_path, "orig_1to1", {"n_ids": 10}, _toy_1to1(), _toy_1toN(), {"mismatch": 0})
    assert res["json"].is_file()
    names = {p.name for p in res["csvs"]}
    assert "orig_1to1_roc.csv" in names
    assert "orig_1to1_cmc.csv" in names
    assert "orig_1to1_score_dist.csv" in names
    # JSON валиден и детерминирован при повторе.
    j1 = res["json"].read_text(encoding="utf-8")
    write_report(tmp_path, "orig_1to1", {"n_ids": 10}, _toy_1to1(), _toy_1toN(), {"mismatch": 0})
    assert res["json"].read_text(encoding="utf-8") == j1