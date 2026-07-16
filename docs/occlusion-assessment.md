# Окклюзия (маска/очки) — робастность к освещению

**Контекст.** Quality-gate (`app/ml/quality/image_quality_gate.py::_check_occlusion`)
детектирует маску и солнцезащитные очки → `status=retry, reason=remove_occlusion`
(«снимите маску/очки»), ДО passive liveness. Это security-gate (hard-reject, не
смягчается soft-режимом). Документ фиксирует метрики и валидацию на реальных сериях.

## Метрики (робастные к освещению)

Обе метрики **относительные** к самому лицу: эталон-зона берётся из заведомо
открытой части того же лица → сдвиг освещения одинаково влияет на эталон и
проверяемую зону → метрика стабильна. Фиксированные HSV-пороги (прежняя
`lower_face_skin_frac` с `V≥40`) ломались на тусклом/боковом свете (кожа=0 →
ложная маска) — заменены на относительные.

### Mask — `lower_face_v_ratio`

```
v_ratio = mean_V(нижняя зона лица) / median_V(эталон-переносица)
```

- **Эталон-зона**: горизонтальная полоса между глазами и носом (переносица) —
  открыта и при маске, и при очках. Медиана HSV-`V` (робастна к теням/выбросам).
- **Проверяемая зона**: нос → 0.95·h, ширина рот ± 0.15·mouth_dx.
- **Порог** `QUALITY_MIN_LOWER_FACE_V_RATIO = 0.50`: ниже → `mask_detected`.
- **Safe-fail**: `ref_v` None или `<1e-3` → `v_ratio=None`, `mask_detected=False`
  (не ложный брак; детектор отказывается, а не клеймит).
- `V` (brightness) игнорирует hue — главная поломка старой skin-frac на тусклом
  свете устранена.

### Sunglasses — `eye_dark_ratio` (HSV V)

```
eye_dark_ratio = mean_V(eye_band) / mean_V(cheek_band)
```

- **eye_band**: `y ∈ [eye_y - 0.6·half, eye_y + 0.6·half]`, `half = max(2, int(eye_dist·0.22))`.
- **cheek_band**: `y ∈ [eye_y + 1.2·half, eye_y + 2.0·half]` (скулы под глазами).
- **Порог** `QUALITY_MAX_EYE_DARK_RATIO = 0.60`: ниже → `sunglasses_detected`.
- Тёмные солнцезащитные линзы затемняют eye-band относительно щёк → ratio < 1.
- **sat_drop ОПРОБОВАН И ОТМЕНЁН**: гипотеза «добавить `sat_drop = eye_S/cheek_S`
  (CONJ)» провалилась — на реальных очках `sat_drop > 1` (насыщенность глаз не
  падает), не разделяет. Оставлен `eye_dark_ratio` один.

### Glasses (обычные) — `eye_edge_density`

Оправа даёт кра́я в eye-band → Canny edge-density. Порог
`QUALITY_MAX_EYE_EDGE_DENSITY = 0.25`. Гладкая тёмная линза (солнцезащитные) не
ловится edge-density → для неё отдельный `eye_dark_ratio`.

## Валидация на реальных сериях (`scripts/check_series.py`)

Полный `pipeline.process` на сериях (кроп из original full-res, debug-логи убраны):

| Серия | Условие | Результат | mask ложных |
|-------|---------|-----------|-------------|
| **Camera Roll1** (18 чистых) | **разное освещение** (brightness 115–172, боковой/контровой/тусклый) | ok=10, retry=2 (ложные sunglasses тень), spoof=3 (liveness на тёмных/смазанных — НЕ occ), error=3 (Low confidence RetinaFace) | **0** |
| Camera Roll 1-10 (clean) | 1 освещение | 10/10 ok | 0 |
| Camera Roll 11-20 (sunglasses) | 1 освещение | retry 7-8/8 (очки ловятся, 1 среднепрозрачные пропущены) | — |
| Camera Roll 21-30 (mask) | 1 освещение | retry 9/9 (v_ratio 0.31–0.41) | — |

**Главная бага устранена:** прежняя `lower_face_skin_frac` давала 8/18 ложных
`mask_detected` на чистых лицах при тусклом/боковом свете (mfrac 0.0–0.48). Новая
`v_ratio` → **mask ложных = 0** на Camera Roll1.

**Разделение классов** (original-кроп):
- `v_ratio`: чистые 0.586–1.11, маска 0.31–0.41 — зазор без перекрытия, порог 0.50.
- `eye_dark_ratio` (HSV V): чистые 0.589–0.883, очки 0.437–0.654 — порог 0.60
  ловит 7-8/8 очков; 1-2 ложных sunglasses-retry на чистых с глубокой тенью от
  брови (мягкий компромисс: лучше пропустить очки — passive liveness ловит, чем
  браковать 80% чистых).

## Компромиссы и trade-off

1. **Mask safe-fail может пропустить маску**: если эталон-переносица ненадёжна
   (мало пикселей / пересвечена) → `mask_detected=False`. Safe-fail в пользу
   легального пользователя (не ложный брак), но брешь для spoof. Компенсируется
   passive liveness (MiniFASNet ловит маску отдельно) + `occ_min_face_side`
   (мелкие кропы skip).

2. **1-2 ложных sunglasses-retry на чистых-тенях**: тень от надбровной дуги на
   eye-band → `eye_dark < 0.60`. Лучше перебраковать (retry = «снимите очки»,
   пере-снимите) чем ослабить порог и пропустить солнцезащитные. Liveness-порог
   0.859 не трогается (подтверждён серией).

3. **spoof=3 на Camera Roll1** (кадры 5,15,17, live 0.712–0.754) — это passive
   liveness на тёмных/смазанных кадрах, **НЕ occ-проблема**. Liveness-порог 0.859
   подтверждён (см. [liveness-assessment](liveness-accuracy-assessment.md)).

## Что НЕ закрыто (future)

- **Маска/очки при разном освещении** — валидированы на 1 освещении (Camera Roll
  11-30). Серия маска/очки при разном свете — future (нужна доп. серия от оператора).
- `scripts/diag_occ.py` — диагностика HSV-сигналов 3 классов (clean/sunglasses/mask),
  распределения ref/lower/eye/cheek H/S/V + v_ratio + eye_dark + frac_H.

## Артефакты

- `app/ml/quality/image_quality_gate.py::_check_occlusion` — mask v_ratio + sunglasses eye_dark.
- `app/core/config.py` — `QUALITY_MIN_LOWER_FACE_V_RATIO=0.50`, `QUALITY_MAX_EYE_DARK_RATIO=0.60`.
- `scripts/check_series.py` — прогон pipeline на серии кадров.
- `scripts/diag_occ.py` — диагностика HSV-сигналов 3 классов.
- `tests/unit/test_image_quality_gate.py` — 22 unit-теста (включая
  `test_mask_v_ratio_robust_to_illumination` — регрессия на главную багу).
- Memory: `sunglasses-dark-eyes-retry`, `no-capture-compensation`,
  `crop-from-original`, `dark-eyes-calibration-script`.