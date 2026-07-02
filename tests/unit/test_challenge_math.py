# test_challenge_math.py — unit-тесты геометрии/детекторов active-liveness challenge.
#
# Чистая математика (EAR, yaw, pitch, smile, IoU, 3D-consistency, verify_challenge_stream,
# sample_actions) — без моделей/Redis/БД. Маркер 'unit' отключает миграции и flush Redis
# (см. conftest._all_unit). Настройки берём из settings (дефолты config.py).
from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

from app.core.config import settings
from app.ml.liveness.challenge import (
    ACTIONS,
    FrameObservation,
    _bbox_iou,
    _check_3d_consistency,
    _detect_blink,
    _detect_nod,
    _detect_smile,
    _detect_turn,
    _ear_for_eye,
    _mouth_width_ratio,
    _pitch_signal_from_5pt,
    _yaw_from_5pt,
    sample_actions,
    verify_challenge_stream,
)

_unit = pytest.mark.unit


def _lm5(le, re, nose, ml, mr) -> np.ndarray:
    return np.asarray([le, re, nose, ml, mr], dtype=np.float32)


def _obs(
    n: int,
    *,
    ear: float | None = 0.30,
    yaw: float = 0.0,
    pitch: float = 0.50,
    mouth: float = 0.30,
    passive: float = 1.0,
    bbox=(100.0, 100.0, 200.0, 200.0),
) -> list[FrameObservation]:
    """n однородных наблюдений (состояние «покой») — стабильная последовательность."""
    return [
        FrameObservation(
            idx=i, bbox=tuple(bbox), lm106=None,
            lm5=_lm5((40, 50), (60, 50), (50, 60), (45, 80), (55, 80)),
            yaw=yaw, pitch_signal=pitch, ear=ear,
            mouth_width_ratio=mouth, passive_score=passive, ts_ms=float(i) * 100.0,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Геометрия
# ---------------------------------------------------------------------------

@_unit
def test_ear_open_vs_closed_eye():
    """Открытый глаз → большее h/w, чем закрытый (радиус self-calibrating не зависит от индексов)."""
    eye_center = np.asarray([50.0, 50.0])
    radius = 10.0
    # 8 точек вокруг глаза: X в [42,58] (w=16), Y в [47,53] (h=6) — все в радиусе 10
    open_pts = np.asarray(
        [[42, 47], [58, 47], [42, 53], [58, 53], [50, 47], [50, 53], [46, 50], [54, 50]],
        dtype=np.float32,
    )
    # закрытый глаз: Y в [49,51] (h=2), X тот же (w=16)
    closed_pts = np.asarray(
        [[42, 49], [58, 49], [42, 51], [58, 51], [50, 49], [50, 51], [46, 50], [54, 50]],
        dtype=np.float32,
    )
    ear_open = _ear_for_eye(open_pts, eye_center, radius)
    ear_closed = _ear_for_eye(closed_pts, eye_center, radius)
    assert ear_open is not None and ear_closed is not None
    assert ear_open > ear_closed, f"open EAR {ear_open} должен быть > closed {ear_closed}"
    assert abs(ear_open - 6.0 / 16.0) < 1e-6
    assert abs(ear_closed - 2.0 / 16.0) < 1e-6


@_unit
def test_ear_none_when_too_few_points_or_narrow():
    """<4 точек в радиусе или w<2 → None (защита от деградации)."""
    eye_center = np.asarray([50.0, 50.0])
    few = np.asarray([[49, 50], [51, 50]], dtype=np.float32)  # 2 точки
    assert _ear_for_eye(few, eye_center, 10.0) is None
    narrow = np.asarray([[49.2, 50], [49.4, 50], [49.6, 50], [49.8, 50]], dtype=np.float32)  # w<2
    assert _ear_for_eye(narrow, eye_center, 10.0) is None


@_unit
def test_yaw_symmetric_is_zero_asymmetric_nonzero():
    le, re, nose_y = (40.0, 50.0), (60.0, 50.0), 55.0
    sym = _yaw_from_5pt(_lm5(le, re, (50.0, nose_y), (45, 80), (55, 80)))
    assert abs(sym) < 1e-6, "симметричный нос → yaw≈0"
    right = _yaw_from_5pt(_lm5(le, re, (55.0, nose_y), (45, 80), (55, 80)))  # нос к re
    assert right > 0.0, f"нос смещён к re → yaw>0, got {right}"
    # clips to [-90,90]; экстремальная асимметрия не падает
    extreme = _yaw_from_5pt(_lm5(le, re, (100.0, nose_y), (45, 80), (55, 80)))
    assert -91.0 <= extreme <= 91.0


@_unit
def test_pitch_signal_bounds_and_nose_movement():
    """0 — нос на уровне глаз, 1 — на уровне рта; монотонно по nose_y."""
    le, re, ml, mr = (40.0, 50.0), (60.0, 50.0), (45.0, 80.0), (55.0, 80.0)
    s_low = _pitch_signal_from_5pt(_lm5(le, re, (50.0, 50.0), ml, mr))  # нос на уровне глаз
    s_mid = _pitch_signal_from_5pt(_lm5(le, re, (50.0, 65.0), ml, mr))
    s_high = _pitch_signal_from_5pt(_lm5(le, re, (50.0, 80.0), ml, mr))  # нос на уровне рта
    assert abs(s_low) < 1e-6
    assert abs(s_high - 1.0) < 1e-6
    assert s_low < s_mid < s_high


@_unit
def test_mouth_width_ratio():
    lm = _lm5((40, 50), (60, 50), (50, 60), (45, 80), (55, 80))  # ширина рта 10
    assert abs(_mouth_width_ratio(lm, (40, 40, 80, 80)) - 10.0 / 40.0) < 1e-6
    assert _mouth_width_ratio(lm, (0, 0, 0, 0)) == 0.0  # fw<1 → 0


@_unit
def test_bbox_iou_identical_disjoint_partial():
    assert _bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert _bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)
    assert _bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25.0 / 175.0)


# ---------------------------------------------------------------------------
# Детекторы действий
# ---------------------------------------------------------------------------

@_unit
def test_detect_blink_dip_and_recovery():
    base = 0.30
    dip = base * (1.0 - settings.LIVENESS_EAR_DIP_RATIO) * 0.5  # явно ниже порога
    obs = _obs(3, ear=base)
    obs.append(replace(obs[0], ear=dip))
    obs += _obs(3, ear=base)
    assert _detect_blink(obs) is True


@_unit
def test_detect_blink_flat_is_false():
    assert _detect_blink(_obs(8, ear=0.30)) is False  # нет провисания
    assert _detect_blink(_obs(3, ear=0.30)) is False  # <4 точек EAR


@_unit
def test_detect_turn_excursion():
    obs = _obs(3, yaw=0.0)
    obs.append(replace(obs[0], yaw=settings.LIVENESS_YAW_MIN_DEG + 5.0))
    obs += _obs(3, yaw=0.0)
    ok, direction = _detect_turn(obs)
    assert ok is True
    assert direction in {"left", "right"}


@_unit
def test_detect_turn_no_excursion_is_false():
    ok, direction = _detect_turn(_obs(8, yaw=2.0))
    assert ok is False
    assert direction == ""


@_unit
def test_detect_nod_excursion():
    obs = _obs(3, pitch=0.50)
    obs.append(replace(obs[0], pitch_signal=0.50 + settings.LIVENESS_PITCH_MIN_EXCURSION + 0.05))
    obs += _obs(3, pitch=0.50)
    assert _detect_nod(obs) is True
    assert _detect_nod(_obs(8, pitch=0.50)) is False  # плоский


@_unit
def test_detect_smile_delta():
    obs = _obs(3, mouth=0.30)
    obs.append(replace(obs[0], mouth_width_ratio=0.30 + settings.LIVENESS_SMILE_DELTA + 0.06))
    obs += _obs(3, mouth=0.30)
    assert _detect_smile(obs) is True
    assert _detect_smile(_obs(8, mouth=0.30)) is False  # нет прироста


# ---------------------------------------------------------------------------
# 3D-consistency (анти jump-cut/replay)
# ---------------------------------------------------------------------------

@_unit
def test_consistency_stable_is_true():
    assert _check_3d_consistency(_obs(8)) is True


@_unit
def test_consistency_too_few_frames_is_false():
    assert _check_3d_consistency(_obs(settings.LIVENESS_MIN_FRAMES - 1)) is False


@_unit
def test_consistency_jump_cut_is_false():
    """Телепорт лица (disjoint bbox между соседними кадрами) → IoU<порога → False."""
    obs = _obs(4, bbox=(100.0, 100.0, 200.0, 200.0))
    obs += _obs(4, bbox=(500.0, 500.0, 600.0, 600.0))  # disjoint → IoU=0
    assert _check_3d_consistency(obs) is False


@_unit
def test_consistency_area_jitter_is_false():
    """Скачок размера bbox (плоский экран/loop) → CV площади > порога → False."""
    obs = _obs(3, bbox=(0.0, 0.0, 100.0, 100.0))     # area 10000
    obs += _obs(3, bbox=(0.0, 0.0, 200.0, 200.0))     # area 40000 → CV=0.6
    assert _check_3d_consistency(obs) is False


# ---------------------------------------------------------------------------
# verify_challenge_stream — сводный вердикт
# ---------------------------------------------------------------------------

def _blink_obs(n_frames: int = 8) -> list[FrameObservation]:
    """Последовательность с выполненным морганием + стабильными bbox + passive=1."""
    base = 0.30
    dip = base * (1.0 - settings.LIVENESS_EAR_DIP_RATIO) * 0.5
    obs = _obs(3, ear=base)
    obs.append(replace(obs[0], ear=dip))
    obs += _obs(n_frames - 4, ear=base)
    return obs


@_unit
def test_verify_all_actions_plus_consistency_is_live():
    obs = _blink_obs(8)
    # smile тоже выполним
    obs[4] = replace(obs[4], mouth_width_ratio=0.30 + settings.LIVENESS_SMILE_DELTA + 0.06)
    res = verify_challenge_stream(obs, actions=["blink", "smile"])
    assert res.is_live is True
    assert res.reason == "ok"
    assert res.consistency_ok is True
    assert res.actions_performed["blink"] is True
    assert res.actions_performed["smile"] is True
    # score = 0.5*1 + 0.3*1 + 0.2*1 = 1.0
    assert res.confidence == pytest.approx(1.0, abs=1e-6)


@_unit
def test_verify_too_few_frames():
    res = verify_challenge_stream(_obs(3), actions=["blink"])
    assert res.is_live is False
    assert res.reason == "too_few_frames"
    assert res.n_frames == 3


@_unit
def test_verify_consistency_fail_overrides_actions():
    """Jump-cut → consistency_fail, даже если действие выполнено."""
    obs = _blink_obs(8)
    # сломаем трек: последний кадр teleport
    obs[-1] = replace(obs[-1], bbox=(500.0, 500.0, 600.0, 600.0))
    res = verify_challenge_stream(obs, actions=["blink"])
    assert res.is_live is False
    assert res.reason == "consistency_fail"
    assert res.consistency_ok is False


@_unit
def test_verify_actions_incomplete():
    """Consistency ok, но blink не выполнен → actions_incomplete:blink."""
    res = verify_challenge_stream(_obs(8, ear=0.30), actions=["blink"])
    assert res.is_live is False
    assert res.reason.startswith("actions_incomplete:blink")
    assert res.actions_performed["blink"] is False


@_unit
def test_verify_below_threshold_when_passive_low():
    """Все действия + consistency, но passive=0 → score=0.5+0+0.2=0.7 < 0.859 → не live."""
    obs = [replace(o, passive_score=0.0) for o in _blink_obs(8)]
    res = verify_challenge_stream(obs, actions=["blink"])
    assert res.is_live is False
    assert res.reason == "below_threshold"
    assert res.confidence == pytest.approx(0.7, abs=1e-6)


# ---------------------------------------------------------------------------
# sample_actions — не более одного turn-направления
# ---------------------------------------------------------------------------

@_unit
def test_sample_actions_never_two_turns():
    """На любых seed — не более одного из {turn_left, turn_right} в наборе."""
    for seed in range(200):
        pick = sample_actions(rng=random.Random(seed))
        assert 1 <= len(pick) <= settings.LIVENESS_CHALLENGE_ACTIONS
        assert all(a in ACTIONS for a in pick)
        turns = [a for a in pick if a in {"turn_left", "turn_right"}]
        assert len(turns) <= 1, f"seed={seed} pick={pick} — два turn-направления"


@_unit
def test_sample_actions_turn_appears_sometimes():
    """turn должен появляться с разумной частотой (не вырожден в never-turn баг)."""
    n_with_turn = sum(
        1 for seed in range(200)
        if any(a in {"turn_left", "turn_right"} for a in sample_actions(rng=random.Random(seed)))
    )
    # ожидаем ~половину (один turn в кандидатах из 4, выборка 2 из 4)
    assert n_with_turn >= 60, f"turn появляется лишь {n_with_turn}/200 — Regression: never-turn bug"


@_unit
def test_sample_actions_deterministic_with_seed():
    assert sample_actions(rng=random.Random(123)) == sample_actions(rng=random.Random(123))