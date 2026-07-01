# tests/evaluation/test_datasets.py — unit-тесты datasets + embed_cache (FS на tmp_path).

import json

import numpy as np
import pytest

from evaluation.datasets import (
    flatten,
    iter_id_images,
    load_folder_dataset,
    dataset_stats,
)
from evaluation.embed_cache import (
    cache_key,
    cache_paths,
    get_or_build,
    load,
    save,
)

pytestmark = pytest.mark.unit


# ---------------- datasets ---------------------------------------------------

def _make_dataset(tmp_path):
    root = tmp_path / "ds"
    (root / "id_a").mkdir(parents=True)
    (root / "id_b").mkdir(parents=True)
    (root / "id_a" / "2.jpg").write_bytes(b"")
    (root / "id_a" / "1.jpg").write_bytes(b"")
    (root / "id_a" / "note.txt").write_bytes(b"")  # не-изображение — игнор
    (root / "id_b" / "1.png").write_bytes(b"")
    (root / "empty").mkdir()  # папка без изображений — пропускается
    return root


def test_load_folder_dataset_sorted_and_filtered(tmp_path):
    root = _make_dataset(tmp_path)
    ds = load_folder_dataset(root)
    assert list(ds.keys()) == ["id_a", "id_b"]  # отсортировано по id
    assert [p.name for p in ds["id_a"]] == ["1.jpg", "2.jpg"]  # отсортировано по файлу
    assert len(ds["id_b"]) == 1


def test_iter_id_images_order(tmp_path):
    root = _make_dataset(tmp_path)
    ds = load_folder_dataset(root)
    seq = [(id_, p.name) for id_, p in iter_id_images(ds)]
    assert seq == [
        ("id_a", "1.jpg"),
        ("id_a", "2.jpg"),
        ("id_b", "1.png"),
    ]


def test_flatten_parallel(tmp_path):
    root = _make_dataset(tmp_path)
    ds = load_folder_dataset(root)
    files, ids = flatten(ds)
    assert len(files) == len(ids) == 3
    assert ids == ["id_a", "id_a", "id_b"]


def test_dataset_stats(tmp_path):
    root = _make_dataset(tmp_path)
    ds = load_folder_dataset(root)
    stats = dataset_stats(ds)
    assert stats["n_ids"] == 2
    assert stats["n_images"] == 3
    assert stats["min_per_id"] == 1
    assert stats["max_per_id"] == 2
    assert stats["n_single_image_ids"] == 1  # id_b


def test_load_folder_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_folder_dataset(tmp_path / "nope")


# ---------------- embed_cache ------------------------------------------------

def test_cache_key_and_paths(tmp_path):
    assert cache_key("orig", 512) == "orig_det512"
    npz, meta = cache_paths(tmp_path, "extracted", 0)
    assert npz.name == "extracted_det0.npz"
    assert meta.name == "extracted_det0.meta.json"


def test_cache_save_load_roundtrip(tmp_path):
    emb = np.eye(4, dtype=np.float32)
    ids = ["a", "a", "b", "c"]
    files = ["x/1.jpg", "x/2.jpg", "y/1.jpg", "z/1.jpg"]
    meta = {"source_root": "/data", "model_sha256": "abc123", "created_at": "2026-07-01T00:00:00"}
    save(tmp_path, "orig", 512, emb, ids, files, meta)

    loaded = load(tmp_path, "orig", 512)
    assert loaded is not None
    assert loaded["embeddings"].shape == (4, 4)
    assert loaded["ids"] == ids
    assert loaded["files"] == files
    assert loaded["meta"]["slug"] == "orig"
    assert loaded["meta"]["det_size"] == 512
    assert loaded["meta"]["n"] == 4
    assert loaded["meta"]["model_sha256"] == "abc123"
    # meta-файл — валидный JSON с ключами.
    meta_path = cache_paths(tmp_path, "orig", 512)[1]
    parsed = json.loads(meta_path.read_text(encoding="utf-8"))
    assert parsed["embedding_dim"] == 4


def test_cache_load_missing(tmp_path):
    assert load(tmp_path, "orig", 512) is None


def test_cache_get_or_build_hit(tmp_path):
    emb = np.ones((2, 3), dtype=np.float32)
    save(tmp_path, "orig", 320, emb, ["a", "b"], ["f1", "f2"], {"created_at": "x"})
    calls = {"n": 0}

    def builder():
        calls["n"] += 1
        return (emb, ["a", "b"], ["f1", "f2"], {"created_at": "x"})

    res = get_or_build(tmp_path, "orig", 320, builder)
    assert calls["n"] == 0  # кеш-хит → builder не вызывался
    assert res["ids"] == ["a", "b"]


def test_cache_get_or_build_miss(tmp_path):
    calls = {"n": 0}

    def builder():
        calls["n"] += 1
        emb = np.eye(3, dtype=np.float32)
        return (emb, ["a", "b", "c"], ["f1", "f2", "f3"], {"created_at": "y", "source_root": "/d"})

    res = get_or_build(tmp_path, "extracted", 0, builder)
    assert calls["n"] == 1
    assert res["embeddings"].shape == (3, 3)
    # повторный вызов — кеш-хит, builder не дёргается.
    res2 = get_or_build(tmp_path, "extracted", 0, builder)
    assert calls["n"] == 1
    assert res2["ids"] == ["a", "b", "c"]


def test_cache_get_or_build_force(tmp_path):
    save(tmp_path, "orig", 512, np.zeros((2, 2), np.float32), ["a", "b"], ["f1", "f2"], {"created_at": "old"})
    calls = {"n": 0}

    def builder():
        calls["n"] += 1
        return (np.ones((2, 2), np.float32), ["a", "b"], ["f1", "f2"], {"created_at": "new"})

    res = get_or_build(tmp_path, "orig", 512, builder, force=True)
    assert calls["n"] == 1  # force → пересборка
    assert np.all(res["embeddings"] == 1.0)
    assert res["meta"]["created_at"] == "new"