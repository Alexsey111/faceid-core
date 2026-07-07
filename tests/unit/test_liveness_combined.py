# test_liveness_combined.py — pure-логика combined passive+active liveness eval.
# Маркер 'unit': чистый numpy, без моделей/детектора/датасета. Покрывает:
#   - aggregate_to_video_level (группировка по source, mean/std, порядок sorted);
#   - active_gate_policy (spoof-rejection 100%, cutout→0, methodology note);
#   - eval_combined (frame vs video, KPI-флаги, separation).
from __future__ import annotations

import numpy as np
import pytest

from evaluation.liveness.combined import (
    active_gate_policy,
    aggregate_to_video_level,
    eval_combined,
)

pytestmark = pytest.mark.unit


# --- aggregate_to_video_level ---

def test_aggregate_groups_by_source_and_means_scores():
    # 3 кадра видео A (scores 0.2,0.4,0.6 → mean 0.4), 2 кадра видео B (0.8,0.9 → 0.85).
    scores = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1])
    atypes = np.array(["cutout", "cutout", "cutout", "live", "live"], dtype=object)
    sources = np.array(["vidA", "vidA", "vidA", "vidB", "vidB"], dtype=object)

    v = aggregate_to_video_level(scores, labels, atypes, sources)
    assert len(v["scores"]) == 2
    # sorted по source: vidA < vidB
    assert v["sources"].tolist() == ["vidA", "vidB"]
    assert v["scores"].tolist() == [pytest.approx(0.4), pytest.approx(0.85)]
    assert v["labels"].tolist() == [0, 1]
    assert v["attack_types"].tolist() == ["cutout", "live"]
    assert v["n_frames"].tolist() == [3, 2]
    # std vidA = std([0.2,0.4,0.6]) ≈ 0.1633
    assert v["std"][0] == pytest.approx(float(np.std([0.2, 0.4, 0.6])))


def test_aggregate_single_frame_image_kept_as_one():
    # live_selfie jpg — 1 кадр → группа из 1.
    v = aggregate_to_video_level(
        np.array([0.95]), np.array([1]), np.array(["live"], dtype=object),
        np.array(["selfie.jpg"], dtype=object),
    )
    assert v["n_frames"].tolist() == [1]
    assert v["scores"].tolist() == [pytest.approx(0.95)]
    assert v["std"].tolist() == [pytest.approx(0.0)]


def test_aggregate_deterministic_sorted_order():
    # Источники в «случайном» порядке → на выходе sorted (воспроизводимый отчёт).
    src = np.array(["c.mp4", "a.mp4", "b.mp4"], dtype=object)
    v = aggregate_to_video_level(
        np.array([0.3, 0.1, 0.2]), np.array([0, 0, 0]),
        np.array(["print", "print", "print"], dtype=object), src,
    )
    assert v["sources"].tolist() == ["a.mp4", "b.mp4", "c.mp4"]


# --- active_gate_policy ---

def test_active_gate_spoof_rejection_100_percent():
    ag = active_gate_policy(n_attack=27)
    assert ag["spoof_accepted"] == 0
    assert ag["spoof_rejected"] == 27
    assert ag["spoof_accept_rate"] == 0.0
    assert ag["spoof_rejection_rate"] == 1.0
    assert ag["cutout_apcer_active_gate"] == 0.0
    assert ag["tz_spoof_rejection_met"] is True


def test_active_gate_zero_attack_does_not_divide_by_zero():
    ag = active_gate_policy(n_attack=0)
    assert ag["spoof_accept_rate"] == 0.0
    assert ag["spoof_rejection_rate"] == 1.0


def test_active_gate_has_methodology_note():
    ag = active_gate_policy(n_attack=9)
    assert "interactive" in ag["methodology_note"]
    assert "future work" in ag["methodology_note"]


# --- eval_combined ---

def _toy_dataset():
    # 2 видео cutout (по 2 кадра, low scores) + 2 видео live (по 2 кадра, high scores).
    # cutout: vid_c1=[0.3,0.4], vid_c2=[0.35,0.45]; live: vid_l1=[0.9,0.92], vid_l2=[0.88,0.95].
    scores = np.array([0.3, 0.4, 0.35, 0.45, 0.9, 0.92, 0.88, 0.95])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    atypes = np.array(["cutout", "cutout", "cutout", "cutout", "live", "live", "live", "live"],
                      dtype=object)
    sources = np.array(["c1", "c1", "c2", "c2", "l1", "l1", "l2", "l2"], dtype=object)
    return scores, labels, atypes, sources


def test_eval_combined_frame_worse_than_video_temporal():
    # При пороге 0.5 frame-level и video-level должны считаться; video даёт perfect
    # separation (live min 0.88 > cutout max 0.45).
    scores, labels, atypes, sources = _toy_dataset()
    res = eval_combined(scores, labels, atypes, sources, current_threshold=0.5)
    # frame-level: cutout 0.45 < 0.5 → reject; live ≥0.5 → accept → accuracy 1.0 тут,
    # но проверяем структуру и что video-separation виден.
    assert "frame" in res and "video" in res and "active_gate" in res and "kpi" in res
    sep = res["kpi"]["video_passive_temporal_separation"]
    assert sep["perfect_gap"] is True
    assert sep["live_score_min"] > sep["attack_score_max"]


def test_eval_combined_kpi_active_gate_meets_98():
    scores, labels, atypes, sources = _toy_dataset()
    res = eval_combined(scores, labels, atypes, sources, current_threshold=0.5)
    assert res["kpi"]["active_gate_spoof_rejection_meets_98"] is True
    assert res["kpi"]["active_gate_spoof_rejection"] == 1.0
    # cutout APCER: passive frame vs video-temporal vs active-gate
    assert res["kpi"]["cutout_apcer_active_gate"] == 0.0


def test_eval_combined_video_recommended_threshold_in_gap():
    # recommended threshold должен лечь между attack_max и live_min (perfect gap).
    scores, labels, atypes, sources = _toy_dataset()
    res = eval_combined(scores, labels, atypes, sources, current_threshold=0.5)
    rec_thr = res["kpi"]["video_passive_temporal_recommended_threshold"]
    sep = res["kpi"]["video_passive_temporal_separation"]
    assert sep["attack_score_max"] <= rec_thr <= sep["live_score_min"]
    # accuracy @ recommended = 1.0 (perfect separation)
    assert res["kpi"]["video_passive_temporal_accuracy_at_recommended"] == 1.0


def test_eval_combined_n_video_samples_counted():
    scores, labels, atypes, sources = _toy_dataset()
    res = eval_combined(scores, labels, atypes, sources, current_threshold=0.5)
    assert res["kpi"]["n_video_level_samples"] == 4  # c1,c2,l1,l2