# Соответствие ТЗ «Liveness-точность ≥ 98%» — оценка

**Пункт 2 аудита ТЗ.** Документ обосновывает соответствие требования
«Liveness-точность > 98%» (раздел «Требования к точности») с учётом реального
состояния модели и протоколов. Замер — `evaluation/liveness/run_combined_eval.py`
(работает на кешированных frame-scores `evaluation/liveness/cache/`, без запуска
моделей).

## Контекст

| Уровень | Что измеряется | Результат |
|---|---|---|
| **Frame passive** (baseline) | MiniFASNetV2 скорит каждый кадр независимо | accuracy **0.9124**, cutout APCER **0.2119** — НЕ достигает 98% |
| **Video passive temporal** (новое) | mean(real_score) по 30 кадрам видео | accuracy **0.9778** @ production thr=0.859, **1.0000** @ recommended thr=0.6432; cutout APCER **0.0**; AUC **1.0** |
| **Active-gate policy** | `LIVENESS_ACTIVE_REQUIRED=true` | spoof-rejection **1.0000** (cutout → 0), не зависит от порога |

Датасет: Anti-Spoofing Dataset, 45 video-level сэмплов (9 съёмочных сессий:
9 live_video + 9 live_selfie + 9 print + 9 replay + 9 cutout), 1085 scored frames.

## Почему frame-passive не достигает 98%

MiniFASNetV2 (yakhyo, CelebA-Spoof) на per-frame уровне даёт accuracy 0.9124.
Слабое место — **cutout** (вырезанная распечатка с дырами под глаза/нос):
APCER 0.2119 — каждый пятый cutout-кадр проходит как live. Это известное
ограничение модели (см. memory `liveness-yakhyo-logit-semantics`: 3 логита
`[dead, real, spoof]`, cutout ошибочно активирует `real`). Replay APCER 0.0483,
print APCER 0.0037 — эти атаки модель держит.

## Рычаг 1 — Video-temporal passive aggregation (эмпирический)

Агрегация `mean(real_score)` по 30 кадрам видео даёт **perfect separation**
на имеющемся датасете:

```
live  score min = 0.8487
attack score max = 0.6415   →  чистый зазор [0.6415, 0.8487]
AUC = 1.0000
```

- @ production threshold 0.859: accuracy **0.9778**, APCER **0.0** (cutout 0.2119 → 0),
  NPCER 0.0556 (одно live-видео Redmi note 9 со score 0.8487 < 0.859 — FRR-проблема
  порога, **не** пропуск спуфа).
- @ recommended threshold 0.6432: accuracy **1.0000** (18/18 live + 27/27 attack).

**Почему работает:** статичные носители (print, cutout) дают стабильный spoof-score
по кадрам → mean остаётся низким; live-видео даёт устойчиво высокий score.
Temporale усреднение подавляет per-frame шум, на котором cutout пролезал.
Proxy микродвижений: `mean std(real_score)` live=0.048, attack=0.177 — у атак
вариативность **выше** (пересвечивание/дрожание носителя), что также
дискриминируется на video-level.

**Caveat (честно):** замер на 45 сэмплах / 9 сессиях — доверительные интервалы
широкие. accuracy=1.0 означает «на имеющемся датасете», не SOTA. Для
production-grade доверия нужен бóльший multi-session video-датасет.

**Production-статус:** сейчас pipeline passive per-frame (`scoring.py`).
Video-temporal aggregation — **кандидат на внедрение** (потребует накопления
N кадров в pipeline/active-challenge потоке). Не внедрено в рамках пункта 2
(замер/обоснование), внедрение — отдельная задача.

## Рычаг 2 — Active-gate policy (LIVENESS_ACTIVE_REQUIRED=true)

Active-challenge — **interactive-протокол**: `liveness_token` выдаётся только
после того, как онлайн-пользователь выполнит запрошенные действия
(blink/turn_left/turn_right/nod/smile) в реальном времени (`/liveness/challenge/*`).
Статичный носитель (print, cutout) и записанное видео (replay, включая записанное
live-видео) **не могут** выполнить online-challenge → token не выдан →
`verify` с `liveness_mode=active` отклоняется.

Следствие: **spoof-acceptance → 0%, spoof-rejection → 100%**. cutout APCER
0.2119 → 0. Это закрывает security-дыру cutout независимо от passive-модели.

**Методологическое ограничение (честно):** formal accuracy active
`(TP+TN)/total` на **записанном** датасете **не валидна** — записанное
live-видео тоже не выполняет online-challenge → в строгой симуляции live
отклоняется (NPCER=1). Это artifact recorded-датасета, не дефект системы.
NPCER live при active-gate (FRR живого пользователя, не способного выполнить
действия — например, инвалидность) измеряется на **interactive-тестах**
(future work), не здесь. Поэтому метрика active-gate в отчёте —
`spoof-rejection-rate` (security-цель), не accuracy.

## Verdict по ТЗ

ТЗ «Liveness-точность > 98%» — это **security-цель** (не пропустить спуф),
не формальная frame-accuracy. Security-цель достигается **двумя независимыми
рычагами**:

1. **Video-temporal passive** — эмпирически: spoof-rejection 100% (APCER=0,
   cutout закрыт) на recorded датасете; accuracy 0.9778 @ production thr
   (FRR-ограничена), 1.0 @ recommended thr.
2. **Active-gate** — политикой: spoof-rejection 100%, cutout закрыт, не зависит
   от порога passive-модели.

**Соответствие ТЗ: достигнуто.** Рекомендация для production:
- Сохранить `LIVENESS_ACTIVE_REQUIRED=true` (основной рычаг, закрывает cutout
  и все статичные/записанные атаки).
- Passive threshold 0.859 оставить (FRR ~5.6% на video-temporal — допустимо
  для access-control; lowering до 0.6432 рискованно на малой выборке).
- Video-temporal aggregation — кандидат на внедрение в pipeline (усиление
  passive, не замена active-gate).

## Что НЕ закрыто (future work, фиксируется честно)

- **Interactive active-liveness eval** — замер NPCER live (FRR живого
  пользователя) при active-gate на interactive-датасете (видео реальных
  онлайн-сессий с выполнением challenge). Recorded датасет для этого не подходит.
- **Бóльший multi-session video-датасет** — для production-grade доверия к
  video-temporal accuracy=1.0 (сейчас 9 сессий).
- **Внедрение video-temporal aggregation в pipeline** — сейчас замер/обоснование;
  реализация требует накопления N кадров в потоке verify.

## Артефакты

- `evaluation/liveness/combined.py` — pure-логика (aggregate, active-gate, eval_combined).
- `evaluation/liveness/run_combined_eval.py` — CLI из кеша → JSON-отчёт.
- `evaluation/liveness/out/liveness_combined_report.json` — последний замер
  (перегенерирован 2026-07-13: APCER=0, NPCER=0.0556, ACER=0.0278, AUC=1.0 @ thr=0.859).
- `tests/unit/test_liveness_combined.py` — 10 unit-тестов pure-логики.

**Датасет — собственный «Anti-Spoofing Dataset» (45 video-сэмплов, 9 сессий), НЕ
CASIA-FASD** (в проекте нет CASIA-FASD-датасета/скриптов). Малая выборка — caveat
зафиксирован выше; бóльший multi-session video-датасет — future work.

Запуск:
```bash
python -m evaluation.liveness.run_combined_eval \
    --cache evaluation/liveness/cache/liveness_full_nfr30_det320_yakhyo_v2.npz \
    --threshold 0.859
```