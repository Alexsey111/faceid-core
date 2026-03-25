# FaceID Core — Pipeline V2 (Async + Batching + Scaling)

## 📌 Обзор

Данная ветка содержит **оптимизированную production-ready версию FaceID Core**.

В отличие от baseline (`main`), здесь реализованы:

* ✅ Асинхронная обработка (Celery)
* ✅ Очередь задач (/verify_async)
* ✅ Backpressure (контроль нагрузки)
* ✅ Micro-batching для inference
* ✅ Метрики (pipeline, DB, queue)
* ✅ Масштабируемая архитектура

---

## 🧠 Архитектура

### High-level flow

```text
Client → API → Queue → Worker → ML Pipeline → DB → Response
```

---

### ML pipeline (V2)

```text
Image
  → Preprocess
  → Fast detector (cheap)
  → RetinaFace fallback (при необходимости)
  → Batch encoder (ArcFace ONNX)
  → Normalize
  → Search (FAISS / pgvector / CPU fallback)
  → Decision
```

---

## ⚙️ Ключевые компоненты

### 1. Async API

* `/verify_async` — основной endpoint
* Быстро отвечает (task_id)
* Основная работа уходит в worker

---

### 2. Workers (Celery)

* `worker_fast` — основной поток обработки
* `worker_heavy` — резерв / тяжёлые задачи
* pool: **prefork (стабильный режим)**

---

### 3. Micro-batching

* Объединяет несколько запросов в один inference
* Настройки:

```env
EMBED_BATCH_ENABLED=true
EMBED_BATCH_SIZE=8
EMBED_BATCH_TIMEOUT_MS=2
EMBED_BATCH_MAX_WAIT_GUARD_MS=50
```

---

### 4. Backpressure

Контролирует входящий поток:

```text
если estimated_queue_delay > threshold → 429
```

Текущий threshold:

```text
~750 ms
```

---

### 5. Поиск

Layered search:

1. FAISS (in-memory)
2. pgvector (Postgres)
3. CPU fallback

---

### 6. Анти-реплей (Redis)

* Защита от повторных запросов
* Не блокирует pipeline при недоступности Redis

---

## 📊 Метрики

Реализованы:

* `queue_delay_ms`
* `pipeline_ms`
* `detect_ms`
* `encode_ms`
* `db_query_time_ms`

---

⚠️ Важно:

* API `/metrics` доступен
* Worker-метрики требуют отдельного экспорта (Prometheus/Grafana)

---

## 📈 Производительность

### После оптимизаций (пример)

| RATE | avg latency | p95 latency | queue_delay avg | 429   |
| ---- | ----------- | ----------- | --------------- | ----- |
| 2    | ~12 ms      | ~22 ms      | ~<1s            | ~0%   |
| 5    | ~23 ms      | ~100 ms     | ~1–2s           | ~2–5% |
| 8    | ~25 ms      | ~80 ms      | растёт          | ~3%   |

---

### SLA (цель)

```text
queue_delay_ms:
avg < 1000 ms
p95 < 2000 ms
```

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

✔ 45 passed
✔ Отдельная test DB
✔ Alembic миграции автоматически

---

## ⚠️ Текущие ограничения

* Bottleneck: **CPU detector (RetinaFace)**
* Worker может перегружаться при RATE > 5
* SLA пока на границе (degraded режим)
* Worker scaling ограничен одним узлом

---

## 🎯 Где используется

* Production API
* Нагрузочное тестирование
* Scaling эксперименты
* Feature development

---

## 🔄 Отличия от baseline

| Фича            | main | pipeline-v2 |
| --------------- | ---- | ----------- |
| Async           | ❌    | ✅           |
| Очередь         | ❌    | ✅           |
| Batching        | ❌    | ✅           |
| Backpressure    | ❌    | ✅           |
| Метрики         | ❌    | ✅           |
| Масштабирование | ❌    | ✅           |

---

## 🧭 Roadmap

Следующие шаги:

1. Оптимизация detector (уменьшить Retina вызовы)
2. Тюнинг batching (latency vs throughput)
3. Redis cache для embeddings
4. Горизонтальное масштабирование workers
5. Полный мониторинг (Grafana)

---

## 📌 Статус

```text
Production-ready (degraded SLA under load)
```

---

## ❗ Важно

* Эта ветка — **основная для развития**
* Изменения вносить через новые feature-ветки
* Перед правками:

  * запрашивать текущий код
  * соблюдать архитектуру
  * не ломать API-контракты
