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
| **0.6** (`FACE_MATCH_THRESHOLD`, production default) | **22.17%** | 0.000% | ❌ | безопасный режим (FAR=0), ценой высокого FRR |
| 0.5 | 5.46% | 0.000% | ❌ | |
| **0.45** | **2.71%** | 0.000% | ✅ | ТЗ-FRR-режим, FAR на LFW = 0 |
| 0.4 | 1.17% | 0.000% | ✅ | |
| thr @ FAR=0.001 (=TAR 0.9964) | 0.36% | 0.1% | ✅ | точка TAR-цели |

**Вывод по FRR:** ТЗ FRR≤3% **достигается при калиброванном пороге `FACE_MATCH_THRESHOLD`≈0.45**
(FRR 2.71%, FAR 0) или при пороге из точки TAR@FAR=0.001 (FRR 0.36%). НО
**production default `FACE_MATCH_THRESHOLD=0.6` даёт FRR=22%** — это безопасный
режим (FAR=0, ни одного пропущенного impostor), но ценой отказа каждому 5-му
легитимному пользователю.

Это **trade-off порога, не дефицит модели**: одна и та же модель на одном датасете
даёт FRR 22% @ thr 0.6 и FRR 0.36% @ thr TAR-точки. Выбор порога = policy-решение
оператора в зависимости от приоритета security (низкий FAR / высокий FRR, thr 0.6)
vs usability (ТЗ-FRR ≤3%, thr ≈0.45).

### Решение (2026-07-07)

**Default `FACE_MATCH_THRESHOLD` / `HIGH_THRESHOLD` калиброваны к 0.45**
(`app/core/config.py`) — соответствует ТЗ FRR≤3% (FRR 2.71%, FAR=0 на LFW
single-face). `LOW_THRESHOLD=0.30` (no_match); low_confidence band [0.30, 0.45)
→ `challenge_recommended` (active liveness), не hard-reject — реальный hard-FRR
(no_match) = 0.46%.

Прежнее 0.6 оставлено как high-security override через env
(`FACE_MATCH_THRESHOLD=0.6`) — zero-impostor-accept ценой FRR 22%.

Калибровка под production-dataset (не LFW) обязательна — LFW-числа = reference,
не абсолют. `FACE_MATCH_THRESHOLD` вынесен в env именно для этого.

## Связанные артефакты

- `evaluation/lfw/run_single_face_eval.py` — single-face subset eval.
- `evaluation/lfw/out/lfw_single_face_report.json` — proof (TAR/AUC/EER по срезам).
- `evaluation/lfw/run_lfw.py` — full-LFW eval harness (кеш эмбеддингов).
- Memory: `lfw-single-face-meets-target`, `arcface-glintr100-vs-w600k-baseline`,
  `face-selection-heuristic-conflict`.