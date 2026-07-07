# app/ml/detection/face_selection.py — выбор главного лица среди детекций SCRFD/RetinaFace.
#
# Проблема: SCRFD на одном фото может вернуть несколько детекций:
#   1) перекрывающиеся дубли (NMS-survivors) одного лица — крупный бокс часто
#      имеет НИЖЕ confidence и ХУЖЕ локализацию, чем топ-1;
#   2) отдельные фоновые лица/постеры (multi-face) — мелкий фоновый может иметь
#      высокую confidence, и тогда faces[0] (топ-confidence) выберет не того.
# Эмпирика на двух датасетах (замер 2026-07-02):
#   - кропнутые лица (custom 1680-id): faces[0] (pure conf) лучше — largest-area даёт
#     regression TAR@FAR=0.001 0.9842→0.9635 (дубли с ~равной площадью, но худшей
#     локализацией у крупного — largest берёт крупный, битый эмбеддинг);
#   - full-scene (LFW с фоновыми людьми): largest-area лучше (faces[0] берёт
#     высокоуверенный мелкий фоновый → битый эмбеддинг, TAR 0.914→0.953).
# Composite conf×area (гипотеза коммита 2b1a025, реализована здесь) покрывает ОБА:
#   score = confidence × area. На кропнутых (area ~равна) conf доминирует →
#   выбирается highest-conf дубликат (лучшая локализация) → custom 0.9842 сохранён.
#   На full-scene (субъект крупный, фон мелкий) area доминирует → выбирается
#   главный субъект → full-LFW ≈ largest. Pure faces[0] и pure largest каждый
#   проигрывает на одном из датасетов; composite выигрывает на обоих.
# Pipeline:
#   1) сгруппировать детекции по перекрытию (IoU≥_IOU_MERGE) — дубли одного лица;
#   2) в каждой группе оставить highest-confidence (лучше локализованный);
#   3) среди представителей выбрать max(confidence × area).
from __future__ import annotations

import numpy as np  # noqa: F401  (типы/совместимость; вычисления на float)

_IOU_MERGE = 0.5  # порог IoU: детекции выше него считаем дубликатами одного лица


def _area(bbox: list[float] | tuple[float, ...]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = _area(a) + _area(b) - inter
    return inter / ua if ua > 0 else 0.0


def _composite_score(det: dict) -> float:
    """Composite conf×area: на кропнутых доминирует conf, на full-scene — area."""
    return float(det.get("confidence", 0.0)) * _area(det["bbox"])


def select_main_face(faces: list[dict]) -> dict:
    """Выбрать главное лицо среди детекций (composite conf×area под оба сценария).

    Args:
        faces: список детекций RetinaFaceDetector.detect —
            {'bbox':[x1,y1,x2,y2], 'confidence':float, 'landmarks':[[x,y],...]|None}

    Returns:
        выбранная детекция (dict). При одном лице — оно же.

    Эвристика:
        - single-face → тривиально (faces[0]);
        - multi-face → группы по IoU (дубли), представитель = highest-conf,
          финал = max(confidence × area). На кропнутых дубли ~равной площади →
          conf разделяет (custom 0.9842); на full-scene субъект крупнее → area
          разделяет (full-LFW ≈ largest). См. замеры в комментарии модуля.
    """
    if len(faces) == 1:
        return faces[0]

    # 1) группы по IoU (дубли одного лица)
    groups: list[list[dict]] = []
    for f in faces:
        placed = False
        for g in groups:
            if _iou(f["bbox"], g[0]["bbox"]) >= _IOU_MERGE:
                g.append(f)
                placed = True
                break
        if not placed:
            groups.append([f])

    # 2) представитель каждой группы = highest-confidence (лучшая локализация)
    reps = [max(g, key=lambda d: d.get("confidence", 0.0)) for g in groups]

    # 3) среди представителей — max(confidence × area): composite покрывает оба датасета.
    #    confidence как tie-break при ~равной area (нормализация не нужна — area в
    #    пикселях² доминирует при существенной разнице размеров, conf — при равных).
    return max(reps, key=_composite_score)