# test_image_quality_gate.py — unit-тесты новых проверок quality-gate (п.5 аудита):
# равномерность освещения, жёсткая тень, окклюзия (маска/очки → retry «снимите»).
#
# Синтетические np-картинки + 5-pt landmarks; gate конструируется в тесте, режимы
# переключаются напрямую (gate.lighting_mode / *_detect_enabled) — без перечитывания
# settings. Маркер 'unit' → без DB/Redis.
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.ml.quality.image_quality_gate import ImageQualityGate

pytestmark = pytest.mark.unit


# 5-pt landmarks для синтетического лица 200×200 (left_eye, right_eye, nose, mouth_l,
# mouth_r). bbox покрывает лицо.
_LM = np.array([[90, 80], [130, 80], [110, 110], [95, 140], [125, 140]], dtype=np.float32)
_BBOX = [60, 50, 160, 180]


def _skin_bgr(h: int = 200, w: int = 200) -> np.ndarray:
    """Картинка skin-tone (HSV H=10,S=120,V=180) → BGR. Весь кроп — кожа."""
    hsv = np.full((h, w, 3), [10, 120, 180], dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _uniform_gray(h: int = 200, w: int = 200, val: int = 128) -> np.ndarray:
    img = np.full((h, w, 3), val, dtype=np.uint8)
    return img


def _lighting_only_gate() -> ImageQualityGate:
    """Gate с отключённой окклюзией — изолирует lighting-проверки (gray-картинки
    не skin-tone, иначе mask-detection сработал бы раньше lighting)."""
    g = ImageQualityGate()
    g.mask_detect_enabled = False
    g.glasses_detect_enabled = False
    return g


# ---------------------------------------------------------------------------
# Lighting: равномерность + тень
# ---------------------------------------------------------------------------


def test_lighting_uniform_passes():
    gate = _lighting_only_gate()
    img = _uniform_gray(val=128)
    res = gate.evaluate_detection(_BBOX, _LM, image=img)
    assert res.passed is True
    assert res.details["lighting_uniformity"] >= gate.min_lighting_uniformity
    assert "occlusion_flags" in res.details


def test_hard_shadow_soft_mode_is_warning():
    gate = _lighting_only_gate()
    gate.lighting_mode = "soft"
    img = _uniform_gray(val=128)
    img[:, :100] = 30  # левая половина тёмная — жёсткая боковая тень
    img[:, 100:] = 210
    res = gate.evaluate_detection(_BBOX, _LM, image=img)
    # soft → не отбрасывает, но warning в details
    assert res.passed is True
    assert res.details.get("lighting_warning") == "hard_shadow"
    assert res.details["shadow_asymmetry"] > gate.max_shadow_asymmetry


def test_hard_shadow_hard_mode_rejects():
    gate = _lighting_only_gate()
    gate.lighting_mode = "hard"
    img = _uniform_gray(val=128)
    img[:, :100] = 30
    img[:, 100:] = 210
    res = gate.evaluate_detection(_BBOX, _LM, image=img)
    assert res.passed is False
    assert res.reason == "hard_shadow"


def test_bad_lighting_symmetric_nonuniform():
    # Вертикальный градиент (верх тёмный, низ светлый): uniformity низкая
    # (ячейки сверху/снизу сильно различаются), но L/R симметрично → asymmetry ~0
    # → bad_lighting, не hard_shadow.
    gate = _lighting_only_gate()
    gate.lighting_mode = "hard"
    img = _uniform_gray(val=40)
    img[100:, :] = 220  # нижняя половина яркая
    res = gate.evaluate_detection(_BBOX, _LM, image=img)
    assert res.passed is False
    assert res.reason == "bad_lighting"
    assert res.details["shadow_asymmetry"] <= gate.max_shadow_asymmetry


def test_lighting_mode_off_skips_lighting_but_keeps_occlusion():
    # lighting off → lighting-проверки пропущены; occlusion всё равно считается
    # (тумблер независим). Чистое skin-лицо → occlusion не срабатывает, pass.
    gate = ImageQualityGate()
    gate.lighting_mode = "off"
    img = _skin_bgr()
    res = gate.evaluate_detection(_BBOX, _LM, image=img)
    assert res.passed is True
    assert res.details.get("lighting_check_skipped") is True
    assert "occlusion_flags" in res.details
    assert res.details["occlusion_flags"]["mask_detected"] is False


# ---------------------------------------------------------------------------
# Окклюзия: маска → retry (всегда, независимо от режимов)
# ---------------------------------------------------------------------------


def test_mask_clean_face_passes():
    gate = ImageQualityGate()
    img = _skin_bgr()
    res = gate.evaluate_detection(_BBOX, _LM, image=img)
    assert res.passed is True
    occ = res.details["occlusion_flags"]
    assert occ["mask_detected"] is False
    assert occ["lower_face_skin_frac"] is not None and occ["lower_face_skin_frac"] > 0.8


def test_mask_detected_triggers_remove_occlusion_regardless_of_mode():
    gate = ImageQualityGate()
    gate.lighting_mode = "soft"  # даже soft не смягчает окклюзию
    gate.mask_detect_enabled = True
    img = _skin_bgr()
    # Чёрный прямоугольник поверх нижней зоны лица (нос → подбородок).
    img[110:195, 88:132] = 0
    res = gate.evaluate_detection(_BBOX, _LM, image=img)
    assert res.passed is False
    assert res.reason == "remove_occlusion"
    assert res.details["occlusion_flags"]["mask_detected"] is True
    assert res.details["occlusion_flags"]["lower_face_skin_frac"] < gate.min_lower_face_skin_frac


def test_mask_detect_disabled_skips_mask():
    gate = ImageQualityGate()
    gate.mask_detect_enabled = False
    img = _skin_bgr()
    img[110:195, 88:132] = 0  # маска есть, но детекция выключена
    res = gate.evaluate_detection(_BBOX, _LM, image=img)
    assert res.passed is True
    assert res.details["occlusion_flags"]["mask_detected"] is False


def test_glasses_clean_eyes_not_detected():
    gate = ImageQualityGate()
    img = _skin_bgr()  # гладкая кожа — мало краёв в зоне глаз
    res = gate.evaluate_detection(_BBOX, _LM, image=img)
    assert res.passed is True
    occ = res.details["occlusion_flags"]
    assert occ["glasses_detected"] is False


def test_glasses_detected_triggers_remove_occlusion():
    gate = ImageQualityGate()
    img = _skin_bgr()
    # Имитация оправы: чёрно-белые вертикальные полосы в зоне обоих глаз
    # (контраст 0↔255 → Canny уверенно детектит края → высокая edge-density).
    for ex in (90, 130):
        for x in range(ex - 12, ex + 12, 4):
            img[68:92, x:x + 2] = 0
            img[68:92, x + 2:x + 4] = 255
    res = gate.evaluate_detection(_BBOX, _LM, image=img)
    assert res.passed is False
    assert res.reason == "remove_occlusion"
    occ = res.details["occlusion_flags"]
    assert occ["glasses_detected"] is True
    assert occ["eye_edge_density"] > gate.max_eye_edge_density


def test_glasses_detect_disabled_skips_glasses():
    gate = ImageQualityGate()
    gate.glasses_detect_enabled = False
    img = _skin_bgr()
    for ex in (90, 130):
        for x in range(ex - 10, ex + 10, 3):
            img[68:92, x:x + 1] = 255
    res = gate.evaluate_detection(_BBOX, _LM, image=img)
    assert res.passed is True
    assert res.details["occlusion_flags"]["glasses_detected"] is False


def test_occlusion_takes_precedence_over_lighting():
    # Маска + плохой свет: клиенту полезнее «снимите маску» (remove_occlusion),
    # а не «улучшите свет» — окклюзия проверяется раньше lighting.
    gate = ImageQualityGate()
    gate.lighting_mode = "hard"
    img = _skin_bgr()
    img[110:195, 88:132] = 0  # маска
    img[:, :100] = 30  # + тень
    res = gate.evaluate_detection(_BBOX, _LM, image=img)
    assert res.reason == "remove_occlusion"


# ---------------------------------------------------------------------------
# Backward-compat: без image новые проверки пропускаются
# ---------------------------------------------------------------------------


def test_backward_compat_no_image_skips_new_checks():
    gate = ImageQualityGate()
    res = gate.evaluate_detection(_BBOX, _LM)  # без image
    assert res.passed is True
    # occlusion_flags не добавляется, lighting-метрик нет — как было раньше
    assert "occlusion_flags" not in res.details
    assert "lighting_uniformity" not in res.details


def test_face_too_small_still_rejects_without_image():
    gate = ImageQualityGate()
    tiny_bbox = [60, 50, 70, 60]  # 10×10 < min_face_side
    res = gate.evaluate_detection(tiny_bbox, _LM)
    assert res.passed is False
    assert res.reason == "face_too_small"