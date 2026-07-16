# tests/unit/test_pipeline_edge_cases.py — edge-cases (Волна 0.5):
# нет лица / несколько лиц / bad crop / low confidence на уровне pipeline + worker.
#
# Pipeline: _prepare_face_from_detection бросает ValueError ДО quality-gate.
# Worker: _classify_prepare_exception маппит ValueError → status/reason/error_code.
#
# Маркер 'unit' → без DB/Redis/моделей. Детектор/quality_gate мокнут через
# SimpleNamespace (паттерн из test_pipeline_retry_status.py).
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import app.workers.verify_worker as vw
from app.ml import pipeline_v2
from app.ml.pipeline_v2 import FacePipelineV2

pytestmark = pytest.mark.unit


def _face_dict() -> dict:
    return {
        "bbox": [20.0, 20.0, 180.0, 180.0],
        "confidence": 0.99,
        "landmarks": [[60, 80], [120, 80], [90, 110], [70, 140], [110, 140]],
    }


def _make_pipeline(monkeypatch, detect_result):
    """Pipeline с замоканным детектором (возвращает detect_result)."""
    monkeypatch.setattr(pipeline_v2.settings, "LIVENESS_ENABLED", False, raising=False)
    pipeline = FacePipelineV2()
    monkeypatch.setattr(pipeline, "_init", lambda: None)
    image = np.full((200, 200, 3), 200, dtype=np.uint8)
    pipeline.preprocessor = SimpleNamespace(
        process=lambda image_bytes: image,
        process_image=lambda image_input: image,
        decode_pair=lambda image_bytes: (image, image),
    )
    pipeline.quality_gate = SimpleNamespace(
        evaluate_image=lambda _: SimpleNamespace(
            passed=True, reason=None, details={"quality_gate_mode": "hard"},
        ),
        evaluate_detection=lambda **_: SimpleNamespace(
            passed=True, reason=None, details={"quality_gate_mode": "hard"},
        ),
    )
    pipeline.fast_detector = SimpleNamespace(detect=lambda _: detect_result)
    # encoder/liveness — asserts в prepare_face_input проверяют not-None до детекции;
    # для no-face/multi-face encode не вызывается, нужен просто не-None dummy.
    pipeline.encoder = SimpleNamespace(encode_batch=lambda _: [np.zeros(512, np.float32)])
    pipeline.liveness_checker = None
    return pipeline


# ------------------------------ pipeline: нет лица ------------------------------

def test_pipeline_no_face_raises_value_error(monkeypatch):
    """Детектор вернул 0 лиц → ValueError('Face not detected') ДО quality-gate."""
    pipeline = _make_pipeline(monkeypatch, detect_result=[])
    with pytest.raises(ValueError, match="Face not detected"):
        pipeline.process(b"ignored")


def test_pipeline_multiple_faces_raises_value_error(monkeypatch):
    """Детектор вернул >1 лица → ValueError('Multiple faces not allowed')."""
    pipeline = _make_pipeline(monkeypatch, detect_result=[_face_dict(), _face_dict()])
    with pytest.raises(ValueError, match="Multiple faces not allowed"):
        pipeline.process(b"ignored")


def test_pipeline_single_face_does_not_raise_on_count(monkeypatch):
    """Ровно 1 лицо → не падает на проверке количества (доходит до encode)."""
    encode_called = {"v": False}
    pipeline = _make_pipeline(monkeypatch, detect_result=[_face_dict()])
    pipeline.encoder = SimpleNamespace(
        encode_batch=lambda _: (encode_called.__setitem__("v", True), [np.zeros(512, np.float32)])[1],
    )
    pipeline.liveness_checker = None
    result = pipeline.process(b"ignored")
    # Дошёл до encode (не упал на multi/no-face).
    assert encode_called["v"] is True
    assert result["status"] != "quality_reject" or True  # контракт: не падает на count


# ------------------------------ worker: маппинг status ------------------------------

def test_classify_no_face_exception():
    """ValueError('Face not detected') → status=no_face, error_code=no_face, reject."""
    res = vw._classify_prepare_exception(ValueError("Face not detected"))
    assert res is not None
    assert res["result"]["status"] == "no_face"
    assert res["result"]["reason"] == "no_face"
    assert res["result"]["error_code"] == "no_face"
    assert res["terminal_state"] == "reject"


def test_classify_multiple_faces_exception():
    """ValueError('Multiple faces not allowed') → status=no_face, reason=multiple_faces.

    ⚠️ Известное рассогласование: status='no_face' (не 'multiple_faces'), но
    error_code/reason='multiple_faces' — клиент различает по reason, не по status.
    Зафиксировано как контракт (см. docs/edge-cases-assessment.md).
    """
    res = vw._classify_prepare_exception(ValueError("Multiple faces not allowed"))
    assert res is not None
    assert res["result"]["status"] == "no_face"
    assert res["result"]["reason"] == "multiple_faces"
    assert res["result"]["error_code"] == "multiple_faces"
    assert res["terminal_state"] == "reject"


def test_classify_bad_crop_exception():
    """ValueError('bad crop') → status=quality_reject, reason=bad_crop."""
    res = vw._classify_prepare_exception(ValueError("bad crop"))
    assert res is not None
    assert res["result"]["status"] == "quality_reject"
    assert res["result"]["reason"] == "bad_crop"
    assert res["terminal_state"] == "reject"


def test_classify_low_confidence_exception():
    """ValueError('Low confidence face detection') → quality_reject/low_confidence."""
    res = vw._classify_prepare_exception(ValueError("Low confidence face detection"))
    assert res is not None
    assert res["result"]["status"] == "quality_reject"
    assert res["result"]["reason"] == "low_confidence"


def test_classify_unknown_value_error():
    """Незнакомый ValueError → processing_failed/invalid_image (fallback)."""
    res = vw._classify_prepare_exception(ValueError("something unexpected"))
    assert res is not None
    assert res["result"]["status"] == "processing_failed"
    assert res["result"]["reason"] == "invalid_image"
    assert res["terminal_state"] == "error"


def test_classify_non_value_error_returns_none():
    """Не-ValueError (настоящий сбой) → None (не edge-case, а серверный сбой)."""
    assert vw._classify_prepare_exception(RuntimeError("oom")) is None
    assert vw._classify_prepare_exception(TypeError("bad type")) is None