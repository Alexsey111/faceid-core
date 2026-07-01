# tests/evaluation/liveness/test_liveness_dataset.py — unit-тесты сборки датасета (tmp dirs).

import pytest

from evaluation.liveness.dataset import (
    CLASS_MAP,
    build_liveness_samples,
    dataset_stats,
)

pytestmark = pytest.mark.unit


def _make_tree(tmp_path, layout: dict):
    """layout: {dir_name: [filenames]} — создаёт папки и пустые файлы-флаги."""
    for d, files in layout.items():
        sub = tmp_path / d
        sub.mkdir(parents=True, exist_ok=True)
        for f in files:
            (sub / f).write_bytes(b"")


def test_class_map_labels():
    assert CLASS_MAP["live_selfie"][0] == 1
    assert CLASS_MAP["live_video"][0] == 1
    assert CLASS_MAP["printouts"][0] == 0
    assert CLASS_MAP["cut-out printouts"] == (0, "cutout")
    assert CLASS_MAP["replay"] == (0, "replay")


def test_build_samples_images_only(tmp_path):
    _make_tree(tmp_path, {
        "live_selfie": ["a.jpg", "b.png"],
    })
    samples = build_liveness_samples(tmp_path, n_frames_per_video=30)
    assert len(samples) == 2
    assert all(s.is_image for s in samples)
    assert all(s.label == 1 for s in samples)
    assert all(s.attack_type == "live" for s in samples)


def test_build_samples_video_expands_to_n_frames(tmp_path):
    _make_tree(tmp_path, {
        "printouts": ["x.mp4"],
    })
    samples = build_liveness_samples(tmp_path, n_frames_per_video=30)
    assert len(samples) == 30
    assert all(not s.is_image for s in samples)
    assert [s.frame_index for s in samples] == list(range(30))
    assert all(s.label == 0 and s.attack_type == "print" for s in samples)


def test_build_samples_mixed_live_selfie_and_video(tmp_path):
    _make_tree(tmp_path, {
        "live_selfie": ["one.jpg"],       # 1 image
        "live_video": ["v.mp4"],          # 30 frames
    })
    samples = build_liveness_samples(tmp_path, n_frames_per_video=30)
    assert len(samples) == 31
    assert sum(s.is_image for s in samples) == 1
    assert sum(not s.is_image for s in samples) == 30
    assert all(s.label == 1 for s in samples)


def test_build_samples_mov_replay(tmp_path):
    _make_tree(tmp_path, {
        "replay": ["a.MOV", "b.mp4"],
    })
    samples = build_liveness_samples(tmp_path, n_frames_per_video=10)
    assert len(samples) == 20
    assert all(s.attack_type == "replay" and s.label == 0 for s in samples)


def test_build_samples_unknown_dir_ignored(tmp_path):
    _make_tree(tmp_path, {
        "live_selfie": ["a.jpg"],
        "unknown_class": ["z.mp4"],   # не в CLASS_MAP → пропущено
    })
    samples = build_liveness_samples(tmp_path, n_frames_per_video=30)
    assert len(samples) == 1


def test_build_samples_deterministic_order(tmp_path):
    layout = {
        "live_selfie": ["c.jpg", "a.jpg", "b.jpg"],
    }
    _make_tree(tmp_path, layout)
    s1 = build_liveness_samples(tmp_path, 30)
    s2 = build_liveness_samples(tmp_path, 30)
    assert [x.path.name for x in s1] == [x.path.name for x in s2]
    # sorted по имени файла
    assert [x.path.name for x in s1] == ["a.jpg", "b.jpg", "c.jpg"]


def test_build_samples_non_media_ignored(tmp_path):
    _make_tree(tmp_path, {
        "live_selfie": ["a.jpg", "notes.txt", "meta.json"],
    })
    samples = build_liveness_samples(tmp_path, 30)
    assert len(samples) == 1


def test_dataset_stats_counts(tmp_path):
    _make_tree(tmp_path, {
        "live_selfie": ["a.jpg", "b.jpg"],     # 2 live images
        "printouts": ["p.mp4"],                # 30 attack frames
    })
    samples = build_liveness_samples(tmp_path, n_frames_per_video=30)
    st = dataset_stats(samples)
    assert st["n_total"] == 32
    assert st["n_live"] == 2 and st["n_attack"] == 30
    assert st["n_images"] == 2 and st["n_video_frames"] == 30
    assert st["by_class"]["live_selfie"] == 2
    assert st["by_class"]["printouts"] == 30
    assert st["by_attack_type"]["print"] == 30
    assert st["by_attack_type"]["live"] == 2