# FaceID Core — Pipeline V3 (Liveness + Observability)

## 📌 Обзор

Данная ветка расширяет `pipeline-v2` и фиксирует:

* ✅ Встроенный **liveness (anti-spoof)**
* ✅ Production-ready ONNX модель (quantized)
* ✅ Наблюдаемость (latency через logs + worker metrics)
* ✅ Подтверждённый SLA под нагрузкой

---

## 🧠 Архитектура

### High-level flow

```text
Client → API → Queue → Worker → ML Pipeline → DB → Response
```

---

### ML pipeline (V3)

```text
Image
  → Preprocess
  → Fast detector
  → RetinaFace fallback
  → Crop / Align
  → Liveness (ONNX, anti-spoof)
      ↳ если spoof → early exit
  → Batch encoder (ArcFace)
  → Normalize
  → Search (FAISS / pgvector / CPU)
  → Decision
```

---

## ⚙️ Liveness (Anti-Spoof)

### Используемая модель

```text
models/liveness.onnx
(best_model_quantized.onnx)
```

---

### Причины выбора

```text
✔ низкая latency (CPU-friendly)
✔ quantized (минимальная нагрузка)
✔ подходит для single-image сценария
✔ стабильная работа в production
```

---

### Preprocessing

```text
input: 128x128
color: RGB
dtype: float32
```

---

### Output

```text
[real_score, spoof_score]
```

---

### Decision

```text
liveness_passed = real_score > threshold
```

---

### Поведение pipeline

```text
если spoof:
→ pipeline завершается
→ encode НЕ вызывается
→ экономия CPU
```

---

## 📊 Liveness Performance

```text
avg: ~10 ms
p50: ~7.5 ms
p95: <16 ms
```

---

### Относительно encoder

```text
liveness ≈ 8% от encode latency
```

---

### Сравнение

```text
encode avg: ~125 ms
liveness avg: ~10 ms
```

---

## 🎯 Вывод

```text
✔ liveness НЕ является bottleneck
✔ безопасно держать ALWAYS ON
✔ SLA не деградирует
```

---

## 📊 Метрики

### Доступны:

```text
queue_delay_ms
pipeline_ms
detect_ms
encode_ms
liveness_ms (через logs)
```

---

### Важно

```text
liveness_ms считается в worker
не виден в API /metrics
```

---

### Текущий подход

```text
✔ логирование (worker logs)
✔ ручная агрегация p50/p95
```

---

### Причина

```text
Celery prefork ≠ Prometheus multiprocess (из коробки)
```

---

## 🎯 Финальная схема

```text
API:
  лёгкий orchestration слой

Workers:
  replicas = 4
  semaphore = 2
  batch_size = 8
  collect_timeout = 50ms

Runtime:
  ONNX intra/inter = 1/1

Flow:
  sync verify — для low-latency одиночных вызовов
  async verify — для очередного compute-path
```

---

## Encoder Performance (CPU tuned)

```text
Threads: 1/1
```

### Note

```text
1/1 now matches the final runtime target for predictable CPU usage.
```

To scale the async worker locally:

```bash
docker compose up --scale worker=2
```

`worker` does not publish a host port; multiple replicas are meant to be scaled horizontally.

---

## 🚀 Запуск

```bash
docker compose up --build
```

---

## 🧪 Тестирование

```bash
pytest -q
```

---

## 📈 Производительность (после V3)

```text
e2e_latency:
avg ~270 ms
p95 ~540 ms

429:
0%
```

---

## ⚠️ Ограничения

```text
• основной bottleneck — encoder (ArcFace)
• RetinaFace fallback дорогой
• worker scaling пока горизонтально ограничен
• полноценный Prometheus multiprocess не реализован
```

---

## 🧭 Roadmap (V3 → V4)

### 1. Detector optimization (приоритет №1)

```text
уменьшить RetinaFace вызовы
→ рост производительности 20–40%
```

---

### 2. Autoscaling workers

```text
динамическое масштабирование worker_fast
```

---

### 3. Embedding cache (Redis)

```text
ускорение повторных запросов
```

---

### 4. Полный мониторинг

```text
Prometheus multiprocess / sidecar exporter
```

---

## 📌 Статус

```text
Production-ready
SLA соблюдается
Security layer (liveness) включён
```

---

## ❗ Важно

```text
• liveness обязателен (ALWAYS ON)
• изменения через feature-ветки
• перед правками — запрашивать текущий код
• соблюдать существующую архитектуру
```

---

# 🧠 Короткий итог ветки

```text
V2 → scalable pipeline
V3 → secure pipeline (liveness) + validated SLA
```



