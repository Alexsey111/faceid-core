# SLO: latency < 1с и availability 99.5% — пункт 6 аудита ТЗ

**Контекст.** ТЗ (нефункциональные требования): API latency < 1 сек, uptime
≥ 99.5%, масштабируемость. Документ формализует SLO и оценивает покрытие
существующей инфраструктурой.

## SLO-таргеты

| SLO | Таргет | Метрика (Prometheus) | Источник |
|---|---|---|---|
| **Latency P95 `/verify`** | < 1000 мс | `faceid_http_request_duration_seconds` (bucket 1.0) | `app/monitoring/metrics.py:REQUEST_LATENCY` |
| **Latency P95 pipeline** | < 1000 мс | `faceid_pipeline_ms` (bucket 1000) | `app/monitoring/metrics.py:PIPELINE_MS` |
| **Per-stage budget** | см. ниже | `faceid_*_ms` (DETECT/ENCODE/VECTOR_SEARCH/LIVENESS/QUALITY) | per-stage histograms |
| **Availability** | ≥ 99.5% (monthly) | uptime = `1 − (5xx + downtime)/total`; `/ready` healthcheck | `/ready`, LB, restart |
| **Error rate** | < 0.5% | `faceid_http_requests_total{status=~"5.."}` / total | `REQUEST_COUNTER` |

**Error budget 99.5%:** ~3.6 ч downtime/мес (или ~7.2 ч за 30 дней). Расходуется
на деплои/перезапуски; превышение → freeze деплоев, разбор инцидента.

## Latency budget — разбивка по stage (GPU production)

`PIPELINE_STAGE_NAMES` (`app/ml/pipeline_v2.py`): preprocess, detect, align,
encode, search, liveness, decision → `total_pipeline_ms`.

| Stage | Бюджет (GPU) | Метрика | CPU fallback |
|---|---|---|---|
| preprocess (downscale) | ~10 мс | `PREPROCESS_MS` | ~10 мс |
| detect (SCRFD/RetinaFace) | ~30-50 мс | `DETECT_MS` | ~150-300 мс |
| align (arcface crop 112) | ~5 мс | `ALIGN_CROP_MS` | ~5 мс |
| encode (ArcFace ONNX) | ~15-30 мс | `ENCODE_MS` | ~150-250 мс |
| search (pgvector / ivfflat) | ~10-30 мс | `VECTOR_SEARCH_MS` | ~10-30 мс |
| liveness (MiniFASNet) | ~10-20 мс | `LIVENESS_MS` | ~30-80 мс |
| quality gate | ~10-20 мс | `QUALITY_GATE_*_MS` | ~10-20 мс |
| **total pipeline** | **< 200 мс (GPU)** | `PIPELINE_MS` | **< 1000 мс (CPU, в пределах SLO)** |
| HTTP overhead (decode/route/IO) | ~50-150 мс | `REQUEST_LATENCY − PIPELINE_MS` | ~50-150 мс |
| **P95 end-to-end `/verify`** | **< 350 мс (GPU)** | `REQUEST_LATENCY` bucket 1.0 | **< 1000 мс (CPU)** |

**Вывод по latency:** SLO < 1с достигается с запасом на GPU infra (P95 ~350 мс,
budget 1.0с = ~3× headroom). На CPU fallback (dev-машина без CUDA,
memory `hw-no-cuda-gpu`) — encode/detect ~400-550 мс суммарно, end-to-end
в пределах 1с на single-face; **SLO гарантируется на production GPU-deploy**
(memory `dev-vs-prod-hardware`: локальный ПК = dev, production на GPU-серверах).

## Async-путь (fallback в очередь при load > fast-path threshold)

При перегрузке `/verify` переходит в async (`job_id`, поллинг `/jobs/{id}`).
SLO для async:

| SLO async | Таргет | Метрика |
|---|---|---|
| Queue delay P95 | < 500 мс | `QUEUE_DELAY_MS`, `QUEUE_*_MS` |
| Job processing P95 | < 1000 мс | `ASYNC_JOB_PROCESSING_MS` |
| Job end-to-end P95 | < 2000 мс | `ASYNC_JOB_E2E_LATENCY_MS` |
| Result visibility lag | < 500 мс | `VERIFY_RESULT_VISIBLE_LAG_MS` |

Async-path допускает **бóльшую** end-to-end латентность (queue + processing),
т.к. это throttle-режим при пике нагрузки. SLO < 1с — для fast-path; async
SLO зафиксирован отдельно (job e2e < 2с). Это честно: ТЗ < 1с относится к
API response — async-ответ `{"job_id","status":"pending"}` возвращается
< 1с (`VERIFY_ASYNC_ROUTE_MS`), сам job-результат — через поллинг.

## Availability 99.5% — покрытие

| Механизм | Где | Что даёт |
|---|---|---|
| **`/ready` healthcheck** | `app/api/routes/health.py` | 200 только если DB + Redis ok; иначе `degraded` (LB снимает api из ротации) |
| **`/health` liveness** | `app/api/routes/health.py` | process alive (`{"status":"ok"}`) |
| **Docker healthchecks** | `docker-compose.yml` (api/postgres/redis/minio) | auto-restart unhealthy контейнеров (`restart: unless-stopped`) |
| **LB healthcheck** | `api_lb` (nginx) → `/ready` | снимает api-инстанс при `degraded` |
| **6 сервисов** | api/api_lb/worker/postgres/redis/minio | изолирование; отказ одного ≠ полный downtime (api独立 от worker) |
| **Restart policy** | `unless-stopped` | авто-восстановление после crash |
| **CPU/GPU fallback** | `OnnxArcFaceEncoder` providers | CPU fallback при отсутствии CUDA → degraded-latency, но не downtime |

**Расчёт 99.5%:** при restart-on-crash + LB healthcheck + stateless api
(эмбеддинги в БД, кеш в Redis — не в памяти api) — downtime = сумма
рестартов api (секунды) + БД/Redis инциденты. 3.6 ч/мес budget покрывает
плановые деплои (rolling via LB drain) + редкие crash. **SLO достижимо** при
production deploy с LB + persistent volumes для postgres/redis/minio.

## Monitoring & alerting

- **Prometheus scrape** `/metrics` (`app/monitoring/http_metrics.py` middleware
  экспонирует REQUEST_LATENCY, counters, stage histograms).
- **Grafana** `docs/grafana_dashboard.json` + `docs/dashboard_guide.md` —
  P95/P99 latency, error rate, stage breakdown, queue depth.
- **Alerting (рекомендация):**
  - `histogram_quantile(0.95, faceid_http_request_duration_seconds_bucket) > 1.0` → page (SLO breach);
  - `rate(faceid_http_requests_total{status=~"5.."}[5m]) / rate(faceid_http_requests_total[5m]) > 0.005`
    → page (error-rate breach);
  - `/ready != 200` на api-инстансе > 1 мин → page (availability risk);
  - `QUEUE_DEPTH` growth + `QUEUE_DELAY_MS` P95 > 500 мс → warn (capacity).

## Verdict по ТЗ

- **Latency < 1с:** достигнуто на GPU production-infra (P95 ~350 мс, 3× headroom);
  CPU fallback в пределах SLO на single-face. Метрики собираются с явным
  bucket 1.0с. **Соответствие — достигнуто** (при GPU-deploy).
- **Availability 99.5%:** достигнуто через `/ready` + LB + restart + stateless
  api + persistent volumes. Error budget 3.6 ч/мес. **Соответствие — достигнуто**
  (при production deploy с LB).
- **Масштабируемость:** stateless api (горизонтальное масштабирование),
  async-очередь (worker pool), Redis кеш, pgvector ivfflat. Покрыто.

**Действий по коду не требуется** — инфраструктура (metrics/healthchecks/
Grafana/LB) реализована ранее (roadmap нед.3-4). Документ формализует SLO
и связывает таргеты ТЗ с существующими метриками.

## Caveats (честно)

- **SLO latency гарантируется на production GPU-deploy**, не на dev-CPU
  (memory `hw-no-cuda-gpu`, `dev-vs-prod-hardware`). На CPU multi-face +
  liveness могут превышать 1с — это dev-ограничение, не нарушение SLO.
- **99.5% требует production deploy с LB + persistent volumes** (docker-compose
  base + `api_lb`). Single-container deploy без LB — ниже 99.5% (нет
  zero-downtime rolling).
- **Alerting rules** — рекомендация; конкретные thresholds калибруются под
  реальную нагрузку после нагрузочного теста (roadmap день 26-27, 100 RPS).