from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.ml import pipeline_v2
from app.ml.pipeline_v2 import FacePipelineV2

pytestmark = pytest.mark.unit


def test_pipeline_always_returns_canonical_timings(monkeypatch):
    monkeypatch.setattr(pipeline_v2.settings, "LIVENESS_ENABLED", False, raising=False)

    pipeline = FacePipelineV2()
    monkeypatch.setattr(pipeline, "_init", lambda: None)

    image = np.full((200, 200, 3), 255, dtype=np.uint8)

    pipeline.preprocessor = SimpleNamespace(
        process=lambda image_bytes: image,
        process_image=lambda image_input: image,
        decode_pair=lambda image_bytes: (image, image),
    )
    pipeline.quality_gate = SimpleNamespace(
        evaluate_image=lambda _: SimpleNamespace(
            passed=True,
            reason=None,
            details={"quality_gate_mode": "ok"},
        ),
        evaluate_detection=lambda **_: SimpleNamespace(
            passed=True,
            reason=None,
            details={"quality_gate_mode": "ok"},
        ),
    )
    pipeline.fast_detector = SimpleNamespace(
        # RetinaFace-контракт: list-of-dicts с bbox/confidence/landmarks
        # (pipeline_v2._prepare_face_from_detection читает face["bbox"] и др.).
        # Раньше мок возвращал list-of-lists [[x,y,x,y,conf]] — старый
        # FastFaceDetector-контракт, удалён вместе с V1.
        detect=lambda _: [
            {"bbox": [0.0, 0.0, 180.0, 180.0], "confidence": 0.99, "landmarks": None},
        ],
    )
    pipeline.encoder = SimpleNamespace(
        encode_batch=lambda _: [np.asarray([1.0, 0.5, 0.25], dtype=np.float32)],
    )
    pipeline.liveness_checker = None

    result = pipeline.process(b"ignored")
    timings = result["timings"]

    expected_keys = {
        "preprocess_ms",
        "detect_ms",
        "align_ms",
        "encode_ms",
        "search_ms",
        "liveness_ms",
        "decision_ms",
        "total_pipeline_ms",
    }

    assert expected_keys.issubset(timings.keys())
    assert timings["search_ms"] == 0.0
    assert timings["liveness_ms"] == 0.0
    assert isinstance(timings["decision_ms"], float)
    assert timings["total_pipeline_ms"] >= timings["preprocess_ms"]
