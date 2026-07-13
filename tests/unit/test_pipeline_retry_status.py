# test_pipeline_retry_status.py — проводка status="retry" (окклюзия) через pipeline_v2.
#
# quality_gate.evaluate_detection мокнут на reason="remove_occlusion" →
# _prepare_face_from_detection должен вернуть status="retry" (не "quality_reject"),
# а encode/liveness не должны вызваться. Маркер 'unit' → без DB/Redis.
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.ml import pipeline_v2
from app.ml.pipeline_v2 import FacePipelineV2

pytestmark = pytest.mark.unit


def _face_dict() -> dict:
    return {
        "bbox": [20.0, 20.0, 180.0, 180.0],
        "confidence": 0.99,
        "landmarks": [[60, 80], [120, 80], [90, 110], [70, 140], [110, 140]],
    }


def test_pipeline_remove_occlusion_returns_retry(monkeypatch):
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
            passed=True,
            reason=None,
            details={"quality_gate_mode": "hard"},
        ),
        # Окклюзия → блокирующий retry (passed=False, reason=remove_occlusion).
        evaluate_detection=lambda **_: SimpleNamespace(
            passed=False,
            reason="remove_occlusion",
            details={
                "quality_gate_mode": "hard",
                "quality_warning": "remove_occlusion",
                "occlusion_flags": {"mask_detected": True, "glasses_detected": False},
            },
        ),
    )
    pipeline.fast_detector = SimpleNamespace(detect=lambda _: [_face_dict()])

    encode_called = {"v": False}
    pipeline.encoder = SimpleNamespace(
        encode_batch=lambda _: (encode_called.__setitem__("v", True), [np.zeros(512, np.float32)])[1],
    )
    pipeline.liveness_checker = None

    result = pipeline.process(b"ignored")

    assert result["status"] == "retry"
    assert result["quality_reason"] == "remove_occlusion"
    assert result["quality_details"]["occlusion_flags"]["mask_detected"] is True
    # encode/liveness не должны вызываться — лицо не дошло до ArcFace.
    assert encode_called["v"] is False


def test_pipeline_capture_quality_returns_quality_reject(monkeypatch):
    # Контраст: capture-quality (bad_lighting) → status="quality_reject", не retry.
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
            passed=False,
            reason="bad_lighting",
            details={"quality_gate_mode": "hard", "quality_warning": "bad_lighting"},
        ),
    )
    pipeline.fast_detector = SimpleNamespace(detect=lambda _: [_face_dict()])
    pipeline.encoder = SimpleNamespace(encode_batch=lambda _: [np.zeros(512, np.float32)])
    pipeline.liveness_checker = None

    result = pipeline.process(b"ignored")
    assert result["status"] == "quality_reject"
    assert result["quality_reason"] == "bad_lighting"