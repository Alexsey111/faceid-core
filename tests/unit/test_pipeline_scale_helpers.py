# tests/unit/test_pipeline_scale_helpers.py — покрытие модульного хелпера
# scale_faces_to_original: пересчёт bbox/landmarks из координат downscaled-кадра
# (где идёт детекция) в координаты original (откуда берётся кроп лица/occ/embedding/
# liveness). Без моделей/БД — чистая арифметика.

from __future__ import annotations

import numpy as np
import pytest

from app.ml.pipeline_v2 import scale_faces_to_original


@pytest.mark.unit
def test_scale_bbox_and_landmarks_proportionally():
    """original 1024×1024, downscaled 480×480 → sx=sy=1024/480≈2.133.
    bbox и landmarks пересчитываются по обеим осям."""
    orig_shape = (1024, 1024, 3)
    ds_shape = (480, 480, 3)
    sx = sy = 1024 / 480.0

    face = {
        "bbox": [100.0, 200.0, 300.0, 400.0],
        "confidence": 0.95,
        "landmarks": np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
    }
    scaled = scale_faces_to_original([face], orig_shape, ds_shape)

    assert len(scaled) == 1
    s = scaled[0]
    assert s["bbox"] == [
        100.0 * sx, 200.0 * sy, 300.0 * sx, 400.0 * sy,
    ]
    # confidence и прочие поля сохраняются
    assert s["confidence"] == 0.95
    # landmarks масштабированы по осям
    lm = np.asarray(s["landmarks"], dtype=np.float32)
    assert lm[0, 0] == pytest.approx(10.0 * sx)
    assert lm[0, 1] == pytest.approx(20.0 * sy)
    assert lm[1, 0] == pytest.approx(30.0 * sx)
    assert lm[1, 1] == pytest.approx(40.0 * sy)


@pytest.mark.unit
def test_scale_does_not_mutate_input():
    """Хелпер возвращает новый список/словари; исходный face не меняется."""
    orig_shape = (1024, 1024, 3)
    ds_shape = (480, 480, 3)
    bbox_orig = [100.0, 200.0, 300.0, 400.0]
    lm_orig = np.array([[10.0, 20.0]], dtype=np.float32)
    face = {
        "bbox": list(bbox_orig),
        "landmarks": lm_orig.copy(),
    }
    _ = scale_faces_to_original([face], orig_shape, ds_shape)

    assert face["bbox"] == bbox_orig  # не мутирован
    np.testing.assert_array_equal(face["landmarks"], lm_orig)


@pytest.mark.unit
def test_scale_invariant_when_shapes_equal():
    """original == downscaled → sx=sy=1 → координаты не меняются."""
    shape = (480, 640, 3)
    face = {"bbox": [10.0, 20.0, 30.0, 40.0], "landmarks": None}
    scaled = scale_faces_to_original([face], shape, shape)
    assert scaled[0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert scaled[0]["landmarks"] is None


@pytest.mark.unit
def test_scale_independent_axes_non_square():
    """Неквадратный кадр с разным масштабом по осям: 3456×4608 → 360×480.
    sx=3456/360=9.6, sy=4608/480=9.6 — равномерный resize даёт одинаковый масштаб,
    но считаем отдельно (устойчивость к округлению)."""
    orig_shape = (4608, 3456, 3)  # (H, W)
    ds_shape = (480, 360, 3)
    sx = 3456 / 360.0
    sy = 4608 / 480.0
    assert sx == pytest.approx(sy)  # равномерный resize

    face = {"bbox": [10.0, 20.0, 100.0, 200.0]}
    scaled = scale_faces_to_original([face], orig_shape, ds_shape)
    assert scaled[0]["bbox"] == [
        10.0 * sx, 20.0 * sy, 100.0 * sx, 200.0 * sy,
    ]


@pytest.mark.unit
def test_scale_bbox_stays_within_original_bounds_for_typical_detection():
    """bbox детектора на downscaled внутри кадра → после scale попадает в original."""
    orig_shape = (4608, 3456, 3)
    ds_shape = (480, 360, 3)
    # лицо 80×100 на downscaled 360×480
    face = {"bbox": [120.0, 150.0, 200.0, 250.0]}
    scaled = scale_faces_to_original([face], orig_shape, ds_shape)
    x1, y1, x2, y2 = scaled[0]["bbox"]
    assert 0 <= x1 <= 3456 and 0 <= x2 <= 3456
    assert 0 <= y1 <= 4608 and 0 <= y2 <= 4608


@pytest.mark.unit
def test_scale_flat_landmarks_format():
    """Плоский landmarks-массив [x0,y0,x1,y1,...] тоже масштабируется (чётные X,
    нечётные Y) — формат face_align/norm_crop."""
    orig_shape = (1024, 1024, 3)
    ds_shape = (480, 480, 3)
    sx = 1024 / 480.0  # равномерный resize → sx == sy
    lm = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)  # плоский
    face = {"bbox": [0.0, 0.0, 100.0, 100.0], "landmarks": lm}
    scaled = scale_faces_to_original([face], orig_shape, ds_shape)
    out = np.asarray(scaled[0]["landmarks"], dtype=np.float32)
    assert out[0] == pytest.approx(10.0 * sx)   # x0
    assert out[1] == pytest.approx(20.0 * sx)   # y0
    assert out[2] == pytest.approx(30.0 * sx)   # x1
    assert out[3] == pytest.approx(40.0 * sx)   # y1


@pytest.mark.unit
def test_scale_preserves_extra_face_fields():
    """Произвольные поля face (confidence, score и т.д.) копируются как есть."""
    orig_shape = (1024, 1024, 3)
    ds_shape = (480, 480, 3)
    face = {"bbox": [1.0, 2.0, 3.0, 4.0], "confidence": 0.88, "score": 12.5}
    scaled = scale_faces_to_original([face], orig_shape, ds_shape)
    assert scaled[0]["confidence"] == 0.88
    assert scaled[0]["score"] == 12.5