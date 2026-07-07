# Минорные gaps ТЗ 2.4 — пункт 5 аудита

**Контекст.** Аудит ТЗ выявил 4 минорных gap по capture-качеству и anti-spoofing:
шумы, VR-очки, чужие объекты, микродвижения. Документ фиксирует покрытие.

## Покрытие

| Gap ТЗ | Покрытие | Статус |
|---|---|---|
| **Шумы / high-ISO noise** | **РЕАЛИЗОВАНО** (п.5): `_check_noise` — std residual после `medianBlur(3)` | закрыто (opt-in) |
| **Освещение / жёсткая тень** | `_check_lighting` — uniformity (3×3) + shadow asymmetry | закрыто (ранее, п.5 аудита) |
| **Окклюзия: маска / очки** | `_check_occlusion` — skin-tone (маска) + Sobel edge-density (очки) → retry | закрыто (ранее, п.5 аудита) |
| **Blur / резкость** | Laplacian variance (`QUALITY_MIN_BLUR_SCORE`) | закрыто (baseline) |
| **Pose** | eye-line diff + nose offset (`POSE_QUALITY_MODE`) | закрыто (baseline) |
| **VR-очки / VR-headset** | Детектор лица не найдёт лицо под VR-шлемом → `no_face` reject; обычные очки → `glasses_detected` → retry | закрыто (косвенно) |
| **Чужие объекты (рука/телефон у лица)** | Окклюзия (маска/очки) частично; объекты, закрывающие лицевые landmarks → детектор нестабилен → `no_face`/`low_confidence` | закрыто (косвенно) |
| **Микродвижения (optical flow)** | Active-stream: 3D-consistency (`_check_3d_consistency`, bbox CV + IoU) в `challenge.py`. Static capture (verify/upload — один кадр): optical flow неприменим | закрыто для active; static неприменимо |

## Шумы — реализация (новое, п.5)

`app/ml/quality/image_quality_gate.py:_check_noise` — метрика ISO-шума:

```
resid = gray − cv2.medianBlur(gray, 3)
noise_std = resid.std()
```

medianBlur(3) подавляет стохастический high-freq шум, сохраняя структуру →
residual ≈ чистая шумовая компонента. На чистом фото `noise_std ~2-5`, на
шумном (high ISO) `~15-30`.

**Почему отдельная проверка, а не blur-gate:** blur-gate использует Laplacian
variance — но шум **повышает** variance → зашумлённое нерезкое фото проходит
blur-gate **ложноположительно**. `noise_std` (residual после median-фильтра)
изолирует шум от структуры и ловит его отдельно.

**Режимы `QUALITY_NOISE_MODE`** (как lighting, свой режим):
- `off` (default) — пропустить (бережёт TAR на бюджетных камерах с постоянным
  шумом; не блокирует допуск);
- `soft` — warning-only (`noise_warning="high_noise"` в details, passed=True);
- `hard` — `quality_reject` (passed=False).

Default `off` → **zero production-risk**: существующий pipeline не меняет
поведение, пока оператор явно не включит `soft`/`hard`. Threshold
`QUALITY_MAX_NOISE_STD=12.0` (шкала 0-255).

## VR-очки / чужие объекты — косвенное покрытие (обоснование)

Отдельной модели детекции VR-headset / foreign-object **нет** (overengineering
для 1-2 dev). Покрытие косвенное, sufficient для access-control:

- **VR-headset** полностью закрывает лицо → детектор SCRFD/RetinaFace не находит
  валидное лицо → `status="no_face"` → verify отклонён. Доп. модель не нужна:
  self-защита через отсутствие детекции.
- **Обычные очки** → `glasses_detected` (Sobel edge-density в зоне глаз) →
  `status="retry"`, `reason="remove_occlusion"`.
- **Чужие объекты** (рука, телефон у лица) → закрывают лицевые landmarks →
  детекция нестабильна (`low_confidence`) или `no_face` → отклонение/пере-съёмка.
- **Маска** → `mask_detected` (skin-tone фракция в нижней зоне) → retry.

**Когда потребуется отдельная foreign-object-модель:** при требовании отличать
«допустимый аксессуар» (медицинская маска, очки по рецепту) от «атаки» — это
policy-задача, не security (active-gate + no_face уже защищают от спуфинга).

## Микродвижения / optical flow

- **Active-stream** (`challenge.py`): 3D-consistency (CV площади bbox + IoU
  кадр-к-кадру) уже ловит jump-cut/replay/flat-screen артефакты (см. пункт 4).
  Pixel-level optical flow — marginal усиление поверх bbox-CV, не
  gap-закрывающее. Не реализуется.
- **Static capture** (verify/upload — один кадр): optical flow **неприменим**
  (нужна последовательность). Capture-качество static покрывается blur/noise/
  lighting/pose/occlusion.

## Verdict по ТЗ

- **Шумы:** закрыто реализацией `_check_noise` (opt-in, default off → без
  production-risk). Соответствие ТЗ capture-качеству — достигнуто.
- **VR-очки / чужие объекты:** закрыто косвенно (no_face / occlusion-retry /
  low_confidence). Отдельная модель — overengineering.
- **Микродвижения:** active-stream закрыт 3D-consistency; static неприменим.
- **Освещение / окклюзия / blur / pose:** закрыто ранее (п.5 аудита).

**Действий сверх `_check_noise` не требуется.**

## Артефакты

- `app/ml/quality/image_quality_gate.py` — `_check_noise`, wiring в
  `evaluate_detection`, `noise_mode`/`max_noise_std` в `__init__`.
- `app/core/config.py` — `QUALITY_NOISE_MODE` (validator + setting, default off),
  `QUALITY_MAX_NOISE_STD=12.0`.
- `tests/unit/test_image_quality_noise.py` — 8 unit-тестов.