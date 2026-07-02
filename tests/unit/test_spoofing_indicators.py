# test_spoofing_indicators.py — проводка бинарных per-class liveness-вероятностей
# (real_prob=softmax[idx1], spoof_prob=softmax[idx2]) в spoofing_indicators.
#
# Контракт (memory liveness-yakhyo-logit-semantics): yakhyo MiniFASNetV2 имеет 3
# выхода, но idx0 — мёртвый класс, idx1=real, idx2=spoof. Модель бинарная, НЕ
# различает print/replay/cutout — поэтому индикаторы {real_prob, spoof_prob}, а не
# фантомные per-attack метки. Маркер 'unit' → без DB/Redis/модели.
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.ml import pipeline_v2
from app.ml.liveness import scoring as scoring_mod
from app.ml.liveness.onnx_liveness import OnnxLivenessChecker
from app.ml.liveness.scoring import score_image_liveness
from app.ml.pipeline_v2 import FacePipelineV2

pytestmark = pytest.mark.unit


def _make_checker(logits: np.ndarray) -> OnnxLivenessChecker:
    """OnnxLivenessChecker без загрузки реальной ONNX-модели: сессия мокнута."""
    checker = object.__new__(OnnxLivenessChecker)
    checker.REAL_IDX = 1
    checker.input_name = "input"
    checker.input_size = 80
    checker.crop_scale = 2.7
    _logits = np.asarray(logits, dtype=np.float32)

    class _FakeSession:
        def run(self, _outputs, _feed):
            return [_logits]

    checker.session = _FakeSession()
    return checker


def _image() -> np.ndarray:
    return np.full((200, 200, 3), 128, dtype=np.uint8)


# --- OnnxLivenessChecker.predict_probs ---

def test_predict_probs_returns_binary_real_spoof():
    # logits [[2, 5, 1]] → softmax: idx1 (real) доминирует, idx2 (spoof) мал.
    checker = _make_checker([[2.0, 5.0, 1.0]])
    probs, ok = checker.predict_probs(_image(), (50, 50, 150, 150))

    assert ok is True
    assert set(probs.keys()) == {"real", "spoof"}, "idx0 не должен попадать в выход"
    assert probs["real"] > probs["spoof"]
    # real + spoof < 1: часть массы осталась на мёртвом idx0 (он не активируется
    # на реальных кадрах, но здесь синтетический логит — проверяем лишь разбивку).
    assert 0.0 <= probs["real"] <= 1.0
    assert 0.0 <= probs["spoof"] <= 1.0


def test_predict_probs_empty_crop_returns_zero_indicators():
    checker = _make_checker([[2.0, 5.0, 1.0]])
    # bbox с нулевой площадью → crop_face_square вернёт None → ok=False, нули.
    probs, ok = checker.predict_probs(_image(), (0, 0, 0, 0))

    assert ok is False
    assert probs == {"real": 0.0, "spoof": 0.0}


def test_predict_delegates_to_predict_probs():
    checker = _make_checker([[2.0, 5.0, 1.0]])
    probs, _ = checker.predict_probs(_image(), (50, 50, 150, 150))
    real_score, ok = checker.predict(_image(), (50, 50, 150, 150))

    assert ok is True
    assert real_score == probs["real"]


# --- score_image_liveness ---

def _fake_detector(faces):
    return SimpleNamespace(detect=lambda _img: faces)


def test_scoring_includes_spoofing_indicators(monkeypatch):
    checker = _make_checker([[2.0, 5.0, 1.0]])  # real доминирует
    detector = _fake_detector([{"bbox": (50, 50, 150, 150)}])
    # preprocessor.process_image возвращает то же изображение — мокаем.
    monkeypatch.setattr(
        scoring_mod.ImagePreprocessor, "process_image", lambda self, img: img
    )

    result = score_image_liveness(_image(), detector, checker, threshold=0.5)

    assert result["face_detected"] is True
    assert result["liveness"] is True
    assert "spoofing_indicators" in result
    ind = result["spoofing_indicators"]
    assert set(ind.keys()) == {"real_prob", "spoof_prob"}
    assert ind["real_prob"] == result["score"]
    assert ind["spoof_prob"] < ind["real_prob"]


def test_scoring_no_face_zero_indicators(monkeypatch):
    checker = _make_checker([[2.0, 5.0, 1.0]])
    detector = _fake_detector([])  # лиц нет
    monkeypatch.setattr(
        scoring_mod.ImagePreprocessor, "process_image", lambda self, img: img
    )

    result = score_image_liveness(_image(), detector, checker, threshold=0.5)

    assert result["face_detected"] is False
    assert result["spoofing_indicators"] == {"real_prob": 0.0, "spoof_prob": 0.0}


# --- pipeline_v2: liveness_spoof_score в spoof и ok ---

def _face_dict() -> dict:
    return {
        "bbox": [20.0, 20.0, 180.0, 180.0],
        "confidence": 0.99,
        "landmarks": [[60, 80], [120, 80], [90, 110], [70, 140], [110, 140]],
    }


def _pipeline(monkeypatch, liveness_probs: dict) -> FacePipelineV2:
    monkeypatch.setattr(pipeline_v2.settings, "LIVENESS_ENABLED", True, raising=False)
    monkeypatch.setattr(pipeline_v2.settings, "LIVENESS_THRESHOLD", 0.5, raising=False)

    pipeline = FacePipelineV2()
    monkeypatch.setattr(pipeline, "_init", lambda: None)

    image = np.full((200, 200, 3), 200, dtype=np.uint8)
    pipeline.preprocessor = SimpleNamespace(
        process=lambda _b: image,
        process_image=lambda _i: image,
    )
    pipeline.quality_gate = SimpleNamespace(
        evaluate_image=lambda _: SimpleNamespace(
            passed=True, reason=None, details={"quality_gate_mode": "hard"},
        ),
        evaluate_detection=lambda **_: SimpleNamespace(
            passed=True, reason=None, details={"quality_gate_mode": "hard"},
        ),
    )
    pipeline.fast_detector = SimpleNamespace(detect=lambda _: [_face_dict()])
    pipeline.encoder = SimpleNamespace(encode_batch=lambda _: [np.zeros(512, np.float32)])

    class _FakeChecker:
        REAL_IDX = 1

        def predict_probs(self, _img, _bbox):
            return ({"real": liveness_probs["real"], "spoof": liveness_probs["spoof"]}, True)

        def predict(self, _img, _bbox):
            return liveness_probs["real"], True

    pipeline.liveness_checker = _FakeChecker()
    return pipeline


def test_pipeline_spoof_carries_liveness_spoof_score(monkeypatch):
    # real=0.2 < threshold 0.5 → pipeline отдаёт status="spoof" со spoof_score.
    pipeline = _pipeline(monkeypatch, {"real": 0.2, "spoof": 0.8})

    result = pipeline.process(b"ignored")

    assert result["status"] == "spoof"
    assert result["liveness_passed"] is False
    assert result["liveness_score"] == pytest.approx(0.2)
    assert result["liveness_spoof_score"] == pytest.approx(0.8)


def test_pipeline_ok_carries_liveness_spoof_score(monkeypatch):
    # real=0.9 ≥ threshold → ok, оба score проброшены в результат.
    pipeline = _pipeline(monkeypatch, {"real": 0.9, "spoof": 0.1})

    result = pipeline.process(b"ignored")

    assert result["status"] == "ok"
    assert result["liveness_passed"] is True
    assert result["liveness_score"] == pytest.approx(0.9)
    assert result["liveness_spoof_score"] == pytest.approx(0.1)