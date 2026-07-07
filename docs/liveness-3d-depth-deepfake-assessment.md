# 3D-depth и Deepfake-детекция — пункт 4 аудита ТЗ

**Контекст.** ТЗ 2.4 (Liveness / anti-spoofing) подразумевает защиту от
плоских носителей (print/cutout/replay) и синтетических лиц (deepfake).
Документ оценивает: что уже есть, что покрывает ТЗ, что — future work.

## Что уже реализовано (покрывает часть ТЗ 2.4)

### 3D-consistency (`app/ml/liveness/challenge.py:_check_3d_consistency`)
Anti-flat-screen / anti-replay heuristic по последовательности кадров
active-challenge:
- **CV площади bbox** ≤ `LIVENESS_CONSISTENCY_AREA_CV`: плоский экран
  (телефон/планшет) при физическом повороте даёт **скачки** видимой площади
  лица (ракурс экрана → projected area меняется нелинейно), реальное 3D-лицо —
  плавно. CV ловит этот артефакт.
- **IoU bbox кадр-к-кадру** ≥ `LIVENESS_CONSISTENCY_IOU_MIN`: нет «телепорта»
  лица (jump-cut в pre-recorded replay).
- **min frames** гейт.

### 3D-pose (`_yaw_from_5pt`, `_pitch_signal_from_5pt`)
Грубая оценка yaw/pitch по 5pt SCRFD landmarks (arcsin-асимметрия носа /
нормированный pitch-сигнал). Используется для детекции 3D-действий
(turn_left/right, nod) в active-challenge.

### Active-gate 3D-действия (главный рычаг)
`LIVENESS_ACTIVE_REQUIRED=true`: сервер требует online-выполнения случайных
3D-действий (поворот головы, кивок, моргание). **Плоский носитель физически
не может повернуться** → действие не выполнено → `is_live=False` → token не
выдан → verify отклонён. Random nonce + single-use + TTL бьют pre-recorded
replay (запись не адаптируется к nonce). Это закрывает print/cutout/replay
на **политике**, не завися от depth-модели.

## Оценка по типам атак ТЗ 2.4

| Атака | Покрытие сейчас | Уровень |
|---|---|---|
| **Print** (распечатка) | active-gate 3D-action (не повернётся) + passive MiniFASNet (APCER 0.004) | закрыто |
| **Cutout** (вырезанная распечатка) | active-gate (не выполнит action) + video-temporal passive (APCER 0.0, см. пункт 2) | закрыто |
| **Replay** (запись экрана/видео) | random nonce + single-use + TTL + 3D-consistency (bbox CV) | закрыто |
| **Deepfake-replay** (запись синтетики) | random nonce + TTL (запись не адаптируется к challenge) + active-gate (запись не выполнит online action) | закрыто |
| **Live deepfake-stream** (real-time GAN выполняет challenge) | НЕ покрыто — advanced атака | **future work** |
| **3D-mask / silicone** | active-gate частично (texture + 3D-action), passive-texture слаб | частично (low-priority для access-control) |

## Deepfake-детектор — future work (обоснование)

**Отсутствует.** Реализация = отдельная ML-модель (детекция GAN/decoder
artifacts: frequency-domain анализ, EfficientNet на DFDC/FF++-pretrained,
blending-boundary, FPN artifacts). Это **substantial** — отдельная модель,
датасет, eval. Для «1-2 разработчиков, no overengineering» (constraint ТЗ) —
неоправданно, **пока**:

- **Deepfake-replay** (наиболее вероятная атака) уже закрыт active-gate:
  pre-recorded deepfake не выполнит online-challenge с random nonce.
- **Live deepfake-stream** (real-time GAN, адаптируется к challenge) —
  advanced атака, требует реалтайм-генерации высокого качества + согласованности
  с random actions. Уровень threat — выше базового access-control сценария ТЗ.

**Когда потребуется:** при расширении threat-model (high-value target,
zero-trust, public-facing). Тогда — отдельная задача: модель-детектор
deepfake + fusion в `verify_challenge_stream` score (сейчас
`0.5*active + 0.3*passive + 0.2*consistency` → добавить `+ w*deepfake_score`).

## 3D-depth — real monocular depth = overengineering сейчас

**Существующий coverage** (3D-consistency heuristic + pose + active-gate
3D-actions) закрывает flat-носители на security-уровне (см. таблицу выше).

**Real monocular depth** (MiDaS / DepthAnything / UniAD) — отдельная модель,
даёт dense depth-map → flat-носитель детектируется по отсутствию depth-
градиента на лице. Но:
- Отдельная ONNX-модель (~100-400 МБ) + inference latency (~50-100 мс GPU) —
  нарушает SLO <1с (пункт 6) при CPU-fallback (dev-машина без CUDA, memory
  `hw-no-cuda-gpu`).
- Marginal выигрыш над active-gate: active-gate уже закрывает flat-носители
  политикой (3D-action невыполним). Depth-модель дала бы **passive** глубину
  без challenge — удобно UX, но security-выигрыш поверх active-gate минимален.
- Constraint no overengineering.

**Опциональное лёгкое усиление (future, без новой модели):** PnP 3D head-pose
по 5pt/106pt + generic 3D face model (solvePnP, EPnP) → точный yaw/pitch/roll
вместо arcsin-аппроксимации. Усиливает детекцию 3D-действий (точнее экскурсия
yaw), но **не** добавляет flat-screen detection (PnP на плоском носителе даёт
валидный pose — 2D-projection consistent). Marginal для security, не
gap-закрывающее. Не реализуется.

## Verdict по ТЗ 2.4

- **Плоские носители (print/cutout/replay):** закрыто active-gate 3D-action +
  3D-consistency + passive/video-temporal. Соответствие ТЗ — достигнуто.
- **Deepfake-replay:** закрыто active-gate (nonce + TTL + online-action).
  Соответствие — достигнуто для recorded deepfake.
- **Live deepfake-stream:** future work — advanced атака вне базовой threat-model
  ТЗ. Зафиксировано честно, не реализуется в текущем scope (overengineering).
- **Real monocular depth:** overengineering при наличии active-gate;
  existing 3D-consistency heuristic достаточен. Не реализуется.

**Действий по коду не требуется.** Active-gate (`LIVENESS_ACTIVE_REQUIRED=true`)
— основной рычаг; existing 3D-consistency + pose + passive/video-temporal —
defense-in-depth. Deepfake-детектор и real depth — future work при расширении
threat-model. См. [[liveness-combined-eval]], `docs/liveness-accuracy-assessment.md`.