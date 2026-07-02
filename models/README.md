# Models — артефакты ML-моделей

Каталог монтируется в контейнеры (`MODELS_DIR=/app/models`). Структура:

```
models/
  MiniFASNetV2_yakhyo.onnx   # passive liveness (anti-spoof)
  buffalo_l/                 # InsightFace pack: SCRFD det + 2d106det landmarks + w600k_r50 ArcFace
  antelopev2/                # альтернативный ArcFace pack (опционально)
  fast_detector/             # лёгкий детектор для fast-path
  liveness_candidates/       # кандидаты liveness (eval; не используются в runtime)
```

## Passive liveness — `MiniFASNetV2_yakhyo.onnx`

Единственный кандидат, используемый runtime (`app/ml/liveness/model_paths.py`):
`resolve_liveness_model_path` ищет `MiniFASNetV2_yakhyo.onnx`; если файл отсутствует —
возвращает `None`, pipeline/route/task логируют «liveness disabled» (graceful
degradation), падения нет.

Контракт модели (фиксирован в `OnnxLivenessChecker`, guard проверяет 3-класс выход):

| Параметр | Значение |
|---|---|
| Архитектура | MiniFASNetV2 (yakhyo), обучена на CelebA-Spoof |
| Вход | 80×80, BGR, 0–255 (без /255), квадратный кроп `crop_face_square(scale=2.7)` |
| Выход | 3 логита `[dead(idx0), real(idx1), spoof(idx2)]` — эффективно бинарная |
| Порог | `LIVENESS_THRESHOLD = 0.859` (`real = softmax[idx1]`) |
| AUC | 0.9676 на Anti-Spoofing Dataset (eval-регресс) |

Семантика логитов: idx0 — мёртвый класс (не активируется), idx1 = real, idx2 = spoof
(покрывает print + replay). Модель **не различает cutout** (→ real P=0.976) — закрыто
active-challenge gate (`LIVENESS_ACTIVE_REQUIRED`), не моделью. См.
`docs/deploy-runbook.md` § Active liveness gate и memory `liveness-yakhyo-logit-semantics`.

## ArcFace — `buffalo_l/w600k_r50.onnx`

Recognition-энкодер (512D). Смена модели — `settings.ARCFACE_MODEL_REL` (baseline
TAR@FAR=0.9842 на LFW; одно-лицевой LFW достигает 0.9964 после фикса face-selection).
ONNX-провайдер — `settings.ONNX_ARCFACE_PROVIDERS` (`auto` = CUDA→DML→CPU fallback).

## Установка

Модели в репозиторий не коммитытся (крупные бинарники). Положить в `models/` перед
сборкой образа (Dockerfile копирует `models ./models`):

```bash
# liveness
cp MiniFASNetV2_yakhyo.onnx models/
# InsightFace pack (buffalo_l) — скачивается insightface при первом запуске ИЛИ
# кладётся вручную в models/buffalo_l/
docker compose up -d --build
```

Проверка, что liveness подхватился:

```bash
docker compose logs api worker | grep -i "liveness"
# отсутствие "liveness disabled" = модель загружена
```