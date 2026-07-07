# test_face_selection.py — composite conf×area эвристика выбора главного лица.
# Маркер 'unit' → чистая логика, без ML/детектора. Покрывает:
#   - single-face (тривиально);
#   - дубли одного лица (IoU-группировка, highest-conf представитель);
#   - кропнутые дубли ~равной площади → conf разделяет (custom 0.9842-сценарий);
#   - full-scene: субъект крупный + средняя conf vs мелкий фон + высокая conf → субъект;
#   - tie-break и edge (один детектор bbox).
from __future__ import annotations

import pytest

from app.ml.detection.face_selection import (
    _area,
    _composite_score,
    _iou,
    select_main_face,
)

pytestmark = pytest.mark.unit


def _det(bbox, conf):
    return {"bbox": list(bbox), "confidence": conf, "landmarks": None}


# --- _area / _iou ---

def test_area_positive():
    assert _area([0, 0, 10, 20]) == 200.0


def test_area_clamped_negative_dims():
    # вырожденный bbox (x2<x1) → 0, не отрицательное
    assert _area([10, 10, 0, 0]) == 0.0


def test_iou_identical_boxes_is_one():
    assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert _iou([0, 0, 10, 10], [100, 100, 110, 110]) == 0.0


# --- select_main_face ---

def test_single_face_returned_as_is():
    d = _det([0, 0, 100, 100], 0.9)
    assert select_main_face([d]) is d


def test_cropped_duplicates_conf_dominates():
    # Кропнутые дубли одного лица: ~равная площадь, разная confidence.
    # Composite должен выбрать highest-conf (лучшая локализация) — custom-сценарий 0.9842.
    big = _det([0, 0, 112, 112], 0.80)        # крупный, НИЖЕ conf (хуже локализация)
    small = _det([2, 2, 113, 113], 0.97)      # ~та же площадь, ВЫШЕ conf
    # дубли перекрываются (IoU≈0.98) → одна группа, представитель = highest-conf
    chosen = select_main_face([big, small])
    assert chosen is small


def test_full_scene_subject_chosen_over_background():
    # Full-scene: главный субъект крупный + средняя conf, фоновое лицо мелкое + высокое conf.
    # Composite conf×area: субъект 0.88×40000=35200 > фон 0.98×2000=1960 → субъект.
    subject = _det([100, 100, 300, 300], 0.88)
    background = _det([500, 500, 540, 540], 0.98)
    chosen = select_main_face([subject, background])
    assert chosen is subject


def test_iou_grouping_merges_duplicates():
    # Два дубликата одного лица (IoU≥0.5) + отдельное лицо → 2 группы, не 3.
    dup_a = _det([0, 0, 100, 100], 0.9)
    dup_b = _det([5, 5, 105, 105], 0.95)   # IoU с dup_a ≈ 0.81 → одна группа
    other = _det([500, 500, 560, 560], 0.7)  # отдельное
    chosen = select_main_face([dup_a, dup_b, other])
    # представитель группы дубли = dup_b (highest conf), other — отдельная группа.
    # composite: dup_b 0.95×10000=9500 vs other 0.7×3600=2520 → dup_b.
    assert chosen is dup_b


def test_composite_score_formula():
    assert _composite_score(_det([0, 0, 100, 200], 0.5)) == 0.5 * 20000.0


def test_missing_confidence_defaults_to_zero():
    # детекция без confidence → conf=0 → composite=0 (никогда не выбрана среди ненулевых)
    d = {"bbox": [0, 0, 100, 100], "landmarks": None}  # нет confidence
    assert _composite_score(d) == 0.0


def test_tie_break_when_equal_composite():
    # Равная composite (равная conf×area) → max берёт первый в порядке итерации.
    a = _det([0, 0, 100, 100], 0.5)     # 0.5×10000 = 5000
    b = _det([200, 200, 300, 300], 0.5)  # 0.5×10000 = 5000
    chosen = select_main_face([a, b])
    assert chosen in (a, b)


def test_multi_face_subject_with_lower_conf_still_chosen_by_area():
    # Субъект сильно крупнее, но conf заметно ниже фона — area всё равно доминирует.
    # 0.6 × 90000 = 54000 > 0.99 × 2500 = 2475 → субъект. full-scene-край.
    subject = _det([0, 0, 300, 300], 0.60)
    bg = _det([400, 400, 450, 450], 0.99)
    assert select_main_face([subject, bg]) is subject