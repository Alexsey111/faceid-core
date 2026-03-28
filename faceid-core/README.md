# FaceID Core — Async Pipeline Baseline (perf/async-scaling-baseline)

## 📌 Контекст

Данная ветка фиксирует **стабильное baseline-состояние асинхронного pipeline** после устранения искусственных bottleneck'ов и проведения первичного performance-тюнинга.

---

## 🧠 Архитектура

```

Client
→ API (/verify_async)
→ Redis queue
→ Workers (detect → encode → liveness → search)
→ Redis result store
→ API (/jobs/{job_id}/wait)

````

---

## ⚙️ Конфигурация baseline

```text
workers = 4
semaphore = 2
batching = enabled
mode = async
````

---

## 📊 Производительность (single node)

### Тест

* k6 (constant-vus)
* duration: 60s
* endpoint: [http://localhost:8000](http://localhost:8000)

---

### Результаты

| Метрика             | Значение         |
| ------------------- | ---------------- |
| throughput          | ~11.8 jobs/sec   |
| processing_time p95 | ~560 ms          |
| queue_delay p95     | ~6 ms            |
| CPU usage           | ~180% (≈ 2 ядра) |

---

## 📌 Ключевые выводы

### ✅ Очередь больше не bottleneck

```
queue_delay ≈ 6 ms
```

---

### ✅ Pipeline стабилен

```
processing_time p95 ≈ 560 ms
```

---

### ❗ Система CPU-bound

```
CPU ≈ 180–190%
```

---

### ❗ semaphore > 2 не даёт выигрыша

```
sem=2 ≈ sem=3 по throughput
```

---

## 📐 Capacity

```
≈ 12 jobs/sec на одну ноду
```

---

## 🚫 Что сознательно НЕ используется

* aggressive enqueue reject logic
* SLA-based admission
* queue-based throttling

---

## 🔬 Что было оптимизировано

* batch DB operations
* worker concurrency (semaphore)
* batching pipeline
* устранён enqueue bottleneck

---

## ⚠️ Ограничения

* система упирается в CPU
* масштабирование внутри одной ноды ограничено
* LB (api_lb) нестабилен (502), не используется в тестах

---

## 🚀 Дальнейшие шаги

### 1. Горизонтальное масштабирование

```
N nodes → N × 12 jobs/sec
```

---

### 2. Усиление batching

* batch_size → 12 / 16
* проверить влияние на latency и CPU

---

### 3. Оптимизация pipeline

* ускорение detect/encode
* снижение CPU cost

---

### 4. Исправление Load Balancer

* устранить 502
* подготовить production ingress

---

## 📎 Важно

* данная ветка является baseline
* все дальнейшие изменения должны сравниваться с ней
* перед изменениями фиксировать метрики

-