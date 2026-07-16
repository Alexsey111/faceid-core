# Edge-cases: нет лица, несколько лиц, блик — Волна 0.5

**Контекст.** ТЗ (roadmap день 25): «Edge cases (нет лица, несколько лиц, блик)».
Документ фиксирует покрытие этих edge-cases в pipeline + выявленные пробелы.
Замер — research + unit-тесты `tests/unit/test_pipeline_edge_cases.py` (2026-07-13).

## Сводка покрытия

| Edge-case | Реализация | Тесты | Gap |
|-----------|-----------|-------|-----|
| **Нет лица** | ✅ есть | ✅ есть (edge_cases, quality_gate, **edge_cases unit**) | `test_no_face_detected` слабый (только `raises(Exception)`) — дополнен |
| **Несколько лиц** | ✅ есть | ❌ не было → ✅ добавлены (`test_pipeline_edge_cases`) | рассогласование `status` worker vs sync (зафиксировано) |
| **Блик / пересвет** | ❌ **НЕТ явной проверки** | ❌ нет | только global `image_too_bright` (mean>225); локальный блик не ловится |

## 1. Нет лица

**Flow** (`pipeline_v2.py:229-230`):
- Детектор `fast_detector.detect` → 0 лиц → `_prepare_face_from_detection`
  бросает `ValueError("Face not detected")` ДО quality-gate.
- Pre-gate `evaluate_image` (blur/brightness) выполняется раньше детекции
  (`pipeline_v2.py:361`), но пустой/тёмный кадр всё равно дойдёт до детекции.
- Минимальный размер лица: `QUALITY_MIN_FACE_SIDE=72` (soft) / `QUALITY_OCC_MIN_FACE_SIDE=64`
  (hard) — `face_too_small` в quality-gate (`image_quality_gate.py:227-246`),
  hard-reject при <64, soft/hard при <72.

**Маппинг в worker** (`verify_worker.py:334-335`, `_classify_prepare_exception`):
`"no face" in msg or "not detected" in msg` →
`status="no_face"`, `reason="no_face"`, `error_code="no_face"`, `terminal_state="reject"`.

**Маппинг sync** (`verification_service.py:956-965`): `error_code="no_face"`,
`status="processing_failed"`.

**Тесты**: `tests/unit/test_pipeline_edge_cases.py::test_pipeline_no_face_raises_value_error`
+ `test_classify_no_face_exception`. Существующий `tests/test_edge_cases.py:37
test_no_face_detected` слабый (только `pytest.raises(Exception)` без проверки
error_code) — оставлен как smoke на реальном pipeline; контракты покрыты unit.

## 2. Несколько лиц

**Flow** (`pipeline_v2.py:232-233`):
- Детектор возвращает все лица (sorted by confidence, `max_num=0` без лимита).
- `len(faces) > 1` → `ValueError("Multiple faces not allowed")` ДО quality-gate.
- Production жёстко требует ровно 1 лицо (см.
  [recognition-accuracy-assessment](recognition-accuracy-assessment.md): multi-face
  отклоняется → LFW raw 0.9533 — бенчмарк-артефакт, не production-баг).

**⚠️ Рассогласование `status` между путями** (зафиксировано как контракт):
| Путь | `status` | `error_code`/`reason` |
|------|---------|----------------------|
| **Worker** (`verify_worker.py:336-337`) | `no_face` | `multiple_faces` |
| **Sync** (`verification_service.py:960-961`) | `processing_failed` | `multiple_faces` |

Клиент различает «несколько лиц» от «нет лица» по `error_code`/`reason`=
`multiple_faces`, **не** по `status` (worker ставит `no_face` для обоих).
Это有意 designed: worker отдаёт UI-ориентированный `reason`, sync отдаёт
HTTP-семантический `processing_failed`. Менять статус в worker на
`multiple_faces` — ломает UI/демо, в sync — ломает HTTP-контракт. **Оставлено.**

**Тесты**: `test_pipeline_multiple_faces_raises_value_error` +
`test_classify_multiple_faces_exception` (фиксируют контракт `status=no_face` +
`reason=multiple_faces`). До Волны 0.5 **прямых тестов не было** — gap закрыт.

## 3. Блик / засветка (glare / overexposure) — ❌ GAP

**Явной проверки блика НЕТ.** Поиск `glare|overexpos|highlight|specular|пересвет|
блик|засвет` по репо → 0 реализаций (единственное «блик» — комментарий в тесте).

**Что есть** (`image_quality_gate.py:145-151`):
```python
if brightness > self.max_brightness:  # QUALITY_MAX_BRIGHTNESS = 225.0
    return self._wrap_result(passed=False, reason="image_too_bright", ...)
```
- `brightness = float(gray.mean())` — **средняя** яркость всего кадра.
- Локальный блик (specular highlight на части лица: лоб/щёка от вспышки/окна)
  не ловится: mean остаётся <225, лицо частично пересвечено → ArcFace-эмбеддинг
  деградирует, но quality-gate пропускает.

**Почему это пробел (impact):**
- Пересвеченный участок → потеря текстуры → ArcFace-косинус same-person падает →
  ложный `no_match` (FRR растёт). Это зеркало проблемы тёмных кадров
  (см. [occlusion-assessment](occlusion-assessment.md) про illumination-робастность).
- Liveness: пересвет может завысить real-score (модель обучена на ярких live) →
  потенциальный spoof-провод, но менее вероятно чем cutout.

**Что НЕ делать сейчас (out of scope):** внедрение метрики блика = новая фича с
калибровкой (как `eye_dark_ratio`/`v_ratio` — нужен датасет блик-кадров + ROC).
Это отдельная задача, не «фиксация edge-case». Зафиксировано как **future work**.

**Рекомендация (future):** локальная overexposure-метрика:
- `highlight_ratio` = доля пикселей лица с `V > V_hi` (HSV, напр. `V≥250`) AND
  `S < S_lo` (низкая насыщенность = выбеленность). Порог `QUALITY_MAX_HIGHLIGHT_RATIO`.
- Робастно: относительно медианы `V` лица (как `v_ratio` для маски) — блик =
  локальный пик `V` >> медиана. `highlight_excess_ratio = (V > median_V * k)`.
- Threshold → `quality_reject` (не retry — пересвет не «сними что-то», а
  «пересними без вспышки»), reason=`overexposed`.

## Прочие edge-cases (контекст, не входили в ТЗ-тройку)

Quality-gate (`image_quality_gate.py`) уже покрывает:
- `image_blurry` (Laplacian var < `QUALITY_MIN_BLUR_SCORE=45`),
  `image_too_dark` (<`QUALITY_MIN_BRIGHTNESS=35`),
  `image_too_bright` (>225, **global**), `low_contrast` (<18),
  `image_too_small` (<`QUALITY_MIN_IMAGE_SIDE=160`).
- `hard_shadow`/`bad_lighting` (`_check_lighting`, 3×3 grid + L/R asymmetry) —
  ловит **тень/неравномерность**, не пересвет.
- Occlusion: mask/glasses/sunglasses → `remove_occlusion` (см.
  [occlusion-assessment](occlusion-assessment.md)).

**Не покрыто** (future, не в ТЗ-тройке): eye-openness/blink, mouth-open,
head-pitch/yaw в градусах, JPEG-artifacts, hand-over-face.

## Status / error_code — полный список для edge-cases

`VerifyResponse` (`schemas/verify.py:20`): `status`/`error_code`/`reason` —
свободные строки (нет Enum). Значения для edge-cases:

| `status` | `error_code`/`reason` | Когда |
|----------|----------------------|-------|
| `no_face` | `no_face` | 0 лиц (worker) |
| `no_face` | `multiple_faces` | >1 лица (worker) — ⚠️ status не уникален |
| `quality_reject` | `face_too_small` | лицо <64 (hard) / <72 (soft) |
| `quality_reject` | `bad_crop` | roi.mean()<20 / пустой кроп |
| `quality_reject` | `low_confidence` | det confidence < `FAST_CONFIDENCE_THRESHOLD=0.75` |
| `quality_reject` | `image_too_bright` | global mean>225 (**не ловит локальный блик**) |
| `quality_reject` | `image_blurry`/`image_too_dark`/`low_contrast`/`image_too_small` | pre-gate |
| `processing_failed` | `no_face`/`multiple_faces`/`invalid_image` | sync-путь |
| `retry` | `remove_occlusion` | маска/очки (см. occlusion-assessment) |

## Артефакты

- `tests/unit/test_pipeline_edge_cases.py` — 9 unit-тестов: pipeline no-face/
  multi-face/single-face raise + worker `_classify_prepare_exception` маппинг
  (no_face/multiple_faces/bad_crop/low_confidence/unknown/non-ValueError).
- `app/ml/pipeline_v2.py:229-233` — `ValueError("Face not detected")` /
  `ValueError("Multiple faces not allowed")` (ДО quality-gate).
- `app/workers/verify_worker.py:313-354` — `_classify_prepare_exception`
  (маппинг ValueError → status/reason/error_code).
- `app/ml/quality/image_quality_gate.py:145-151` — `image_too_bright` (global,
  единственный сигнал пересвета — НЕ ловит локальный блик).
- Memory: `sunglasses-dark-eyes-retry` (illumination-робастность паттерн),
  `no-capture-compensation` (НЕ компенсировать под dev-камеру),
  `quality-noise-check` (noise-check pattern для новых метрик).

## Вердикт по ТЗ-тройке

- **Нет лица**: ✅ покрыто (pipeline + worker + тесты).
- **Несколько лиц**: ✅ покрыто (pipeline + worker + тести); рассогласование
  `status` зафиксировано как контракт (не баг — UI/HTTP семантика разведена).
- **Блик / пересвет**: ❌ **не покрыто** (future work). Глобальный `image_too_bright`
  не ловит локальный specular-highlight; рекомендация (`highlight_ratio`/relative
  `V`-excess) описана, внедрение = отдельная задача с калибровкой.