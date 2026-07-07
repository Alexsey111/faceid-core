# test_image_quality_noise.py — unit-тесты noise-check (п.5 аудита, минорный gap).
# ISO-шум: std residual после medianBlur(3). Синтетические np-картинки
# (чистая vs зашумлённая) + 5-pt landmarks; gate переключает noise_mode напрямую.
# Маркер 'unit' → без DB/Redis.
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.ml.quality.image_quality_gate import ImageQualityGate

pytestmark = pytest.mark.unit


# 5-pt landmarks для синтетического лица 200×200.
_LM = np.array([[90, 80], [130, 80], [110, 110], [95, 140], [125, 140]], dtype=np.float32)
_BBOX = [60, 50, 160, 180]


def _gate_no_extras() -> ImageQualityGate:
    """Gate с отключёнными occlusion + lighting — изолирует noise-проверку.
    noise_mode ставится в каждом тесте явно (default off в settings)."""
    g = ImageQualityGate()
    g.mask_detect_enabled = False
    g.glasses_detect_enabled = False
    g.lighting_mode = "off"
    g.pose_mode = "off"
    return g


def _clean_face() -> np.ndarray:
    """Гладкое skin-tone лицо (без шума) → низкий noise_std."""
    hsv = np.full((200, 200, 3), [10, 120, 180], dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _noisy_face(seed: int = 0, sigma: float = 40.0) -> np.ndarray:
    """Зашумлённое лицо (high-ISO) — гауссов шум поверх skin-tone → высокий residual."""
    rng = np.random.default_rng(seed)
    img = _clean_face().astype(np.int16)
    img = img + rng.normal(0, sigma, img.shape).astype(np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


# --- _check_noise напрямую ---

def test_noise_off_skips():
    g = _gate_no_extras()
    g.noise_mode = "off"
    r = g._check_noise(_clean_face())
    assert r["passed"] is True
    assert r["details"]["noise_check_skipped"] is True


def test_noise_clean_face_passes_hard():
    g = _gate_no_extras()
    g.noise_mode = "hard"
    g.max_noise_std = 12.0
    r = g._check_noise(_clean_face())
    assert r["passed"] is True
    assert r["details"]["noise_std"] < 12.0


def test_noise_noisy_face_rejects_hard():
    g = _gate_no_extras()
    g.noise_mode = "hard"
    g.max_noise_std = 12.0
    r = g._check_noise(_noisy_face(seed=1, sigma=40.0))
    assert r["passed"] is False
    assert r["reason"] == "high_noise"
    assert r["details"]["noise_std"] > 12.0


def test_noise_noisy_face_soft_is_warning():
    g = _gate_no_extras()
    g.noise_mode = "soft"
    g.max_noise_std = 12.0
    r = g._check_noise(_noisy_face(seed=2, sigma=40.0))
    # soft → passed=True (не отбрасывает), но reason + warning сигнализируют
    assert r["passed"] is True
    assert r["reason"] == "high_noise"
    assert r["details"]["noise_warning"] == "high_noise"


def test_noise_metric_separates_clean_and_noisy():
    # Чистое лицо → noise_std заметно ниже зашумлённого (метрика дискриминативна).
    g = _gate_no_extras()
    g.noise_mode = "hard"
    clean = g._check_noise(_clean_face())["details"]["noise_std"]
    noisy = g._check_noise(_noisy_face(seed=3, sigma=40.0))["details"]["noise_std"]
    assert noisy > clean * 3  # шум даёт кратный рост residual-std


# --- wiring в evaluate_detection ---

def test_evaluate_detection_hard_noise_rejects():
    g = _gate_no_extras()
    g.noise_mode = "hard"
    g.max_noise_std = 12.0
    res = g.evaluate_detection(_BBOX, _LM, image=_noisy_face(seed=4, sigma=40.0))
    assert res.passed is False
    assert res.reason == "high_noise"
    assert res.details["noise_check_mode"] == "hard"


def test_evaluate_detection_soft_noise_passes_with_warning():
    g = _gate_no_extras()
    g.noise_mode = "soft"
    g.max_noise_std = 12.0
    res = g.evaluate_detection(_BBOX, _LM, image=_noisy_face(seed=5, sigma=40.0))
    assert res.passed is True  # soft не отбрасывает
    assert res.details.get("noise_warning") == "high_noise"


def test_evaluate_detection_off_skips_noise():
    g = _gate_no_extras()
    g.noise_mode = "off"
    res = g.evaluate_detection(_BBOX, _LM, image=_noisy_face(seed=6, sigma=40.0))
    assert res.passed is True
    assert res.details.get("noise_check_skipped") is True