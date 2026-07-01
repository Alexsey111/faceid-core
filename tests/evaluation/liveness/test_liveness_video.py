# tests/evaluation/liveness/test_liveness_video.py — unit-тесты индексного сэмплинга (pure).

import pytest

from evaluation.liveness.video import _uniform_indices, is_video

pytestmark = pytest.mark.unit


def test_uniform_indices_basic():
    idx = _uniform_indices(100, 10)
    assert len(idx) == 10
    assert idx[0] == 0 and idx[-1] == 99
    # монотонно неубывающий
    assert all(idx[i + 1] >= idx[i] for i in range(len(idx) - 1))


def test_uniform_indices_capped_at_total():
    # n_frames > total → отдаём total кадров, не больше
    idx = _uniform_indices(5, 30)
    assert idx == [0, 1, 2, 3, 4]


def test_uniform_indices_single_frame_total():
    assert _uniform_indices(1, 30) == [0]


def test_uniform_indices_single_requested():
    # запрошен 1 кадр → первый
    assert _uniform_indices(100, 1) == [0]


def test_uniform_indices_zero_total():
    assert _uniform_indices(0, 30) == []


def test_uniform_indices_zero_requested():
    assert _uniform_indices(100, 0) == []


def test_uniform_indices_unique_when_n_equals_total():
    # n == total → все индексы 0..total-1 без пропусков/дублей
    idx = _uniform_indices(7, 7)
    assert idx == [0, 1, 2, 3, 4, 5, 6]


def test_uniform_indices_endpoint_inclusive():
    # последний индекс строго равен total-1 (covered last frame)
    idx = _uniform_indices(91, 10)
    assert idx[-1] == 90


def test_is_video_extensions():
    assert is_video("a.mp4") and is_video("b.MOV") and is_video("c.AVI")
    assert not is_video("a.jpg") and not is_video("b.png")