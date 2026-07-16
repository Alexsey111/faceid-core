# Recognition-точность и FRR — пункт 1 аудита ТЗ

**Контекст.** ТЗ (раздел «Метрики успеха / ML»): `TAR ≥ 99% @ FAR ≤ 0.1% (LFW)`,
`FRR ≤ 3%`. Документ фиксирует соответствие обоим требованиям на LFW и честно
разводит two-threshold trade-off (TAR/FAR vs FRR при production-пороге).

## Замер (LFW, buffalo_l/w600k_r50, det_size=320, RetinaFace + norm_crop)

Независимый прогон `evaluation/lfw/run_single_face_eval.py` (2026-07-07):
детекция на 7701 уникальных изображениях → face-counts, фильтрация пар где оба
фото ровно 1 лицо, ROC на subset. Артефакт:
`evaluation/lfw/out/lfw_single_face_report.json`.

| Срез пар | n | TAR@FAR=0.001 | TAR@FAR=0.01 | AUC | EER |
|---|---|---|---|---|---|
| ALL (raw LFW) | 6000 | 0.9533 | 0.9553 | 0.9824 | 0.0417 |
| **SINGLE-SINGLE** | 3927 | **0.9964** | 0.9974 | 0.9991 | 0.0036 |
| MULTI-INVOLVED (≥1 multi-face) | 2073 | 0.8724 | 0.8762 | 0.9519 | 0.0979 |

**Production-сценарий = single-face** (селфи = одно лицо). `pipeline_v2.py:191`
жёстко отклоняет multi-face (`Multiple faces not allowed`), на single-face берёт
`faces[0]` (= единственное лицо). Значит production даёт ~0.9964.

MULTI-INVOLVED (0.8724) — **бенчмарк-артефакт** LFW (full-scene кадры с фоновыми
людьми), не дефицит модели: эвристика выбора лица на multi-face берёт фонового →
битый эмбеддинг. В production multi-face не обрабатывается.

## TAR ≥ 99% @ FAR ≤ 0.1% — ✅ достигнуто

SINGLE-SINGLE: **TAR@FAR=0.001 = 0.9964** (цель ≥0.99). AUC 0.9991, EER 0.36%.
Соответствует SOTA ArcFace на LFW. raw-LFW 0.9533 НЕ трактовать как несоответствие
— это full-scene с фоновыми, не «качественный снимок одного лица» по ТЗ.

## FRR ≤ 3% — trade-off порога

FRR на single-single (1958 genuine / 1969 impostor) при фиксированных порогах:

| Порог | FRR | FAR | ТЗ FRR≤3% | Контекст |
|---|---|---|---|---|
| **0.45** (`FACE_MATCH_THRESHOLD`, production default) | **2.71%** | 0.000% | ✅ | operating point: FAR=0≤0.1%, FRR≤3% |
| 0.2634 (thr @ FAR=0.001) | 0.36% | 0.000% | ✅ | точка TAR-цели; лучше UX, но без запаса по FAR для прода |
| 0.5 | 5.46% | 0.000% | ❌ | |
| 0.6 | 22.17% | 0.000% | ❌ | high-security override (FAR=0), ценой FRR 22% |
| **0.73** | **69.10%** | 0.000% | ❌ | ⚠️ КРИТИЧНО — бракует 69% своих (см. ниже) |

**Вывод по FRR:** ТЗ FRR≤3% **достигается при калиброванном пороге `FACE_MATCH_THRESHOLD`=0.45**
(FRR 2.71%, FAR=0). Operating point под чистую FAR≤0.1% = 0.2634 (FRR 0.36%) даёт
лучший UX, но **не применяется**: безопасность-приоритет (СКУД) требует запас против
FAR на проде, плюс LFW переоценивает cosine same-person → реальный FRR при 0.2634
выше 0.36%. 0.45 — сбалансированный operating point с FAR=0 и малым margin до 3%.

### Решение (2026-07-13, перекалибровка)

**Default `FACE_MATCH_THRESHOLD` / `HIGH_THRESHOLD` = 0.45** (`app/core/config.py`)
— соответствует ТЗ (FAR=0≤0.1%, FRR 2.71%≤3% на LFW single-face).
`LOW_THRESHOLD=0.30` (no_match); low_confidence band [0.30, 0.45)
→ `challenge_recommended` (active liveness), не hard-reject.

**Калибровка:** `scripts/calibrate_face_threshold.py` — извлекает operating point
под целевой FAR и печатает FAR/FRR/TAR на наборе порогов-кандидатов (включая
текущие config/.env). LFW-скрипты выбрасывают порог из `tar_at_far`; калибровщик
сохраняет 2-й возврат + `recommend_thresholds`.

**⚠️ Критическая находка (2026-07-13):** локальный `.env` содержал
`FACE_MATCH_THRESHOLD=0.73` (и `FACE_LOW_THRESHOLD=0.60`) — это даёт **FRR=69.1%**
на LFW: бракует ~69% легитимных пользователей. На dev-серии Camera Roll маскировалось
(крупные качественные кропы одного лица дают cosine >0.73 с самим собой), но в проде
с разной позой/светом/камерой 0.73 забракует большинство своих. **Исправлено в .env →
0.45 / 0.30.** Никогда не поднимать порог выше 0.45 без реал-валидации на целевых парах.

`evaluation/protocols.py:CURRENT_HIGH_THRESHOLD` выровнен к 0.45 (был хардкод 0.60,
рассинхрон с config).

### Оговорка: LFW ≠ СКУД

Калибровка на LFW-pairs — **верхняя оценка**. LFW-pairs = «две фото одного/разных»
(похожие условия). Production (СКУД) = эталон при регистрации + кадр с камеры при
проходе (иная поза/свет/возраст/камера) → cosine same-person НИЖЕ → **реальный FRR
в проде выше** LFW-чисел. 0.45 имеет малый margin до 3% — реал-валидация на целевых
парах (когда появятся данные) обязательна, порог может потребовать опускания.
`FACE_MATCH_THRESHOLD` вынесен в env именно для этого.

## Связанные артефакты

- `scripts/calibrate_face_threshold.py` — калибровка operating point под целевой
  FAR (FAR/FRR/TAR на наборе порогов, извлечение thr из tar_at_far).
- `evaluation/lfw/run_single_face_eval.py` — single-face subset eval.
- `evaluation/lfw/out/lfw_single_face_report.json` — proof (TAR/AUC/EER по срезам).
- `evaluation/lfw/run_lfw.py` — full-LFW eval harness (кеш эмбеддингов).
- Memory: `lfw-single-face-meets-target`, `arcface-glintr100-vs-w600k-baseline`,
  `face-selection-heuristic-conflict`.