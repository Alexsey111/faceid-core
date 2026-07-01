# tests/evaluation/test_protocols.py — unit-тесты 1:1 / 1:N протоколов (pure numpy).
# validate_faiss_consistency требует faiss+app → отдельный smoke-прогон (не unit).

import numpy as np
import pytest

from evaluation.protocols import (
    CURRENT_HIGH_THRESHOLD,
    eval_1to1,
    eval_1toN,
)

pytestmark = pytest.mark.unit


def _synth_embeddings(n_ids=6, per_id=3, dim=64, noise=0.05, seed=0):
    """Сепарабельные эмбеддинги: base-вектор на id + малый шум, L2-норм."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_ids, dim)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    E = np.zeros((n_ids * per_id, dim), dtype=np.float32)
    ids = []
    for i, gid in enumerate(np.repeat(np.arange(n_ids), per_id)):
        v = base[gid] + rng.normal(0, noise, dim).astype(np.float32)
        E[i] = v / np.linalg.norm(v)
        ids.append(f"id_{int(gid)}")
    return E, np.array(ids)


def test_eval_1to1_separable_high_tar():
    E, ids = _synth_embeddings()
    r = eval_1to1(E, ids, impostor_ratio=10, seed=42)
    assert "error" not in r
    assert r["n_genuine"] > 0 and r["n_impostor"] > 0
    # Сепарабельные данные → TAR@FAR=0.001 близок к 1, AUC ~ 1.
    assert r["tar_at_far"] >= 0.95
    assert r["auc"] >= 0.99
    assert 0.0 <= r["eer"] <= 0.1


def test_eval_1to1_fields_present():
    E, ids = _synth_embeddings()
    r = eval_1to1(E, ids, impostor_ratio=5, seed=1)
    assert {"n_genuine", "n_impostor", "tar_at_far", "eer", "auc",
            "thresholds", "at_current_high", "roc"} <= set(r.keys())
    rec = r["thresholds"]["recommended"]
    assert {"high", "low", "margin", "eer", "basis"} <= set(rec.keys())
    assert r["thresholds"]["current"]["high"] == CURRENT_HIGH_THRESHOLD
    # roc arrays согласованы по длине.
    far = np.asarray(r["roc"]["far"])
    tar = np.asarray(r["roc"]["tar"])
    assert len(far) == len(tar)


def test_eval_1to1_at_current_high_block():
    E, ids = _synth_embeddings(noise=0.5, seed=3)  # больше шума → не идеально
    r = eval_1to1(E, ids, impostor_ratio=10, seed=7)
    cur = r["at_current_high"]
    assert 0.0 <= cur["far"] <= 1.0
    assert 0.0 <= cur["frr"] <= 1.0
    assert cur["tar"] == pytest.approx(1.0 - cur["frr"])


def test_eval_1to1_determinism():
    E, ids = _synth_embeddings()
    r1 = eval_1to1(E, ids, impostor_ratio=10, seed=42)
    r2 = eval_1to1(E, ids, impostor_ratio=10, seed=42)
    assert r1["tar_at_far"] == r2["tar_at_far"]
    assert r1["auc"] == r2["auc"]
    assert np.array_equal(r1["roc"]["far"], r2["roc"]["far"])


def test_eval_1to1_empty_no_pairs():
    # все id уникальны → genuine нет, impostor ratio*0=0 → нет пар.
    E = np.eye(4, dtype=np.float32)
    ids = np.array(["a", "b", "c", "d"])
    r = eval_1to1(E, ids, impostor_ratio=10, seed=1)
    assert "error" in r


def test_eval_1toN_separable_rank1_one():
    E, ids = _synth_embeddings(n_ids=6, per_id=3, noise=0.02, seed=10)
    r = eval_1toN(E, ids, max_rank=10)
    assert "error" not in r
    assert r["rank1"] == pytest.approx(1.0, abs=1e-6)
    assert r["rank5"] == pytest.approx(1.0, abs=1e-6)
    assert r["n_gallery"] == 6  # по 1 на id
    assert r["n_probes"] == 12  # по 2 остальных на id
    assert r["n_ids_single_image"] == 0


def test_eval_1toN_single_image_ids_gallery_only():
    # id_solo — 1 фото → gallery-only, не queried.
    rng = np.random.default_rng(0)
    base = rng.standard_normal((3, 32)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    # id_a: 3 фото, id_b: 3 фото, id_solo: 1 фото
    E_list = []
    ids_list = []
    for idx, (gid, n) in enumerate([("a", 3), ("b", 3), ("solo", 1)]):
        for _ in range(n):
            v = base[idx] + rng.normal(0, 0.02, 32).astype(np.float32)
            E_list.append(v / np.linalg.norm(v))
            ids_list.append(gid)
    E = np.asarray(E_list, dtype=np.float32)
    ids = np.array(ids_list)
    r = eval_1toN(E, ids, max_rank=5)
    assert r["n_gallery"] == 3  # a, b, solo
    assert r["n_probes"] == 4  # a×2 + b×2 (solo не queried)
    assert r["n_ids_single_image"] == 1  # solo
    assert r["rank1"] == pytest.approx(1.0, abs=1e-5)


def test_eval_1toN_cmc_monotone():
    E, ids = _synth_embeddings(n_ids=8, per_id=2, noise=0.3, seed=5)
    r = eval_1toN(E, ids, max_rank=8)
    acc = np.asarray(r["cmc"]["accuracy"])
    assert np.all(np.diff(acc) >= -1e-9)  # CMC не убывает
    assert acc[-1] == pytest.approx(1.0, abs=1e-6)  # ранг = |gallery| → 1.0