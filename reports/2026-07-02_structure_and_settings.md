# Тестовый отчёт: структура проекта и расположение настроек

- **Дата:** 2026-07-02
- **Проект:** FaceID Core (`C:\Users\Worker\ai\faceid-core`)
- **Ветка:** `main`
- **Тип отчёта:** тестовый (структура + настройки)

---

## 1. Проверка структуры проекта

Корень репозитория — git-корень, приложение лежит в `app/`

### 1.1. Основные директории

| Директория | Назначение |
|---|---|
| `app/` | Приложение FastAPI (API, сервисы, ML-пайплайн, модели БД, инфраструктура-клиенты) |
| `app/api/` | Роутер, зависимости эндпоинтов |
| `app/core/` | Конфиг (`config.py`), логирование, метрики, middleware, крипто (AES) |
| `app/db/` | SQLAlchemy session/base |
| `app/infrastructure/` | Клиенты MinIO, Redis |
| `app/ml/` | ML-пайплайн: `pipeline.py` (v1), `pipeline_v2.py` (v2), `runtime.py`, `batch_encoder.py`, `dependencies.py`, детекция/embedding/liveness/quality |
| `app/models/` | ORM-модели (user, embedding, verification_job, verification_log) |
| `app/monitoring/` | Prometheus-метрики (http, db) |
| `app/schemas/` | Pydantic v2-схемы (upload, verify, liveness, response) |
| `app/services/` | Бизнес-логика: verification, embedding, liveness, faiss, rate_limiter, backpressure, webhook, calibration |
| `app/workers/` | Celery (celery_app, verify_worker) |
| `alembic/` | Миграции БД (`alembic/versions/`) |
| `autoscaler/` | Автоскейлер воркеров |
| `evaluation/` | Eval-harness: `extract.py`, `metrics.py`, `lfw/`, `liveness/`, `cache/` |
| `infrastructure/` | Dockerfile-ы, nginx-конфиги |
| `tests/` | `unit/`, `integration/`, `evaluation/`, `data/`, `data_extended/`, `generated_hard/`, `images/` |
| `scripts/` | Вспомогательные скрипты |
| `docs/` | Документация |
| `models/` | ONNX-модели (`buffalo_l/`, `antelopev2/`, `fast_detector/`, `liveness_candidates/`) |
| `benchmarks/` | Локальные benchmark-артефакты (gitignored) |

### 1.2. Корневые конфиг-файлы

| Файл | Назначение |
|---|---|
| `pyproject.toml` | Зависимости, pytest-конфиг (`pythonpath`), ruff |
| `requirements.txt` | pip-зависимости |
| `alembic.ini` | Конфиг миграций (`script_location = alembic`) |
| `docker-compose.yml` / `docker-compose.node2.yml` | Оркестрация (api, worker, autoscaler, postgres/pgvector, redis, minio, nginx, prometheus) |
| `prometheus.yml` | scrape-конфиг |
| `.env` | Переменные окружения (секреты, **в `.gitignore`**) |
| `.gitignore` | Исключения (модели, кеши, датасеты, мусор бенчмарков) |

### 1.3. Замечания по структуре

- ⚠️ Корень содержит значительный **нагрузочный мусор** (`k6_*`, `summary_*.json`, `cpu_samples_*.csv`, `*.pid`, `queue_*.log`, `workers_*.log`, `uvicorn_*.log`, `metrics_after_*.txt`, `run_*.ps1`) — ~70+ файлов. Часть покрыта `.gitignore`, но физически лежит в корне. Рекомендуется разовая чистка.
- ⚠️ `MODELS_DIR` в `config.py` по умолчанию = `D:/python projects/faceid-core` (устаревший путь開発-машины); в `.env` переопределяется на актуальный.
- ✅ Дубликаты ассетов (`models_backup/`, вложенный `faceid-core/models/buffalo_l/`) отсутствуют — флеттенинг выполнен.

---

## 2. Где лежат настройки

Настройки централизованы в одном классе и читаются из окружения.

### 2.1. Источник настроек

| Слой | Файл / механизм | Описание |
|---|---|---|
| **Схема + дефолты** | `app/core/config.py` → `class Settings(BaseSettings)` | Pydantic v2 `BaseSettings`. Все параметры с дефолтами, валидаторами, alias-ами. Кешируется через `@lru_cache` (`get_settings()`). |
| **Значения env** | `.env` (корень) | Загружается `pydantic-settings` (`SettingsConfigDict`). **В `.gitignore`** — секреты не коммитятся. |
| **Переопределение** | переменные окружения ОС / docker `environment:` | Имеют приоритет над `.env` (для Docker Compose). |
| **DB-URL миграций** | `alembic.ini` → `sqlalchemy.url` | Отдельный от `Settings` путь (синхронный driver для alembic). |
| **Docker** | `docker-compose.yml` | `environment:` и `env_file:` блоки сервисов; порты, volumes. |

### 2.2. Группы настроек в `config.py`

| Группа | Ключевые параметры |
|---|---|
| Application | `ENV`, `APP_ROLE`, `APP_NAME`, `DEBUG`, `APP_VERSION`, `GIT_SHA`, `IMAGE_TAG` |
| PostgreSQL | `POSTGRES_HOST/PORT/DB/USER/PASSWORD`, `DB_HOST`, `DB_POOL_SIZE`, `DB_MAX_OVERFLOW` |
| Redis | `REDIS_HOST/PORT/DB`, `REDIS_ENABLED` |
| MinIO | `MINIO_ENDPOINT/ACCESS_KEY/SECRET_KEY/BUCKET/SECURE` |
| Models | `MODELS_DIR`, `ONNX_INTRA_OP_THREADS`, `ONNX_INTER_OP_THREADS`, `ONNX_ARCFACE_PROVIDERS` (`auto`/`cuda`/`directml`/`cpu`), `ARCFACE_MODEL_REL` |
| Search backend | `SEARCH_BACKEND` (`pgvector`/`faiss`), `FAISS_ENABLED`, `FAISS_INDEX_PATH`, `FAISS_PERSIST_ENABLED` |
| Preprocess | `PREPROCESS_MAX_SIDE` |
| Quality gate | `QUALITY_GATE_MODE` (`hard`/`soft`/`off`), blur/brightness/contrast/face-side, `POSE_QUALITY_MODE`, `RETINA_DET_SIZE` |
| Fast path | `USE_FAST_PATH`, `FAST_PATH_MAX_CONCURRENCY`, `FAST_WORKER_*`, `API_INFER_CONCURRENCY` |
| Liveness | `LIVENESS_ENABLED`, `LIVENESS_THRESHOLD`, `LIVENESS_CROP_SCALE`, `LIVENESS_INPUT_SIZE` |
| Thresholds | `FACE_MATCH_THRESHOLD`, `FACE_LOW_THRESHOLD`, `FACE_MARGIN_THRESHOLD`, `HIGH/LOW/MARGIN_THRESHOLD` |
| Security | `SECRET_KEY`, `AES_SECRET_KEY`, `BIOMETRY_AES_KEY_B64` |
| Auth | `AUTH_ENABLED`, `JWT_ALG/ISSUER/AUDIENCE/SECRET`, `API_KEYS` |
| Webhook | `WEBHOOK_ENABLED/URL/SECRET/TIMEOUT_S/MAX_RETRIES/QUEUE_SIZE/IDEMPOTENCY_TTL_S` |
| Celery / admission | `CELERY_BROKER_URL/RESULT_BACKEND/TASK_QUEUE`, `WORKER_COUNT`, `ASYNC_THROUGHPUT_PER_SEC`, `BACKPRESSURE_*`, `WORKER_SEMAPHORE`, `INFLIGHT_LIMIT`, `MAX_QUEUE_SIZE`, `API_UVICORN_WORKERS` |

### 2.3. Как получить доступ к настройкам в коде

```python
from app.core.config import settings

# примеры
settings.FACE_MATCH_THRESHOLD   # float, 0.6
settings.ONNX_ARCFACE_PROVIDERS # str, "auto"
settings.ARCFACE_MODEL_REL      # str, "buffalo_l/w600k_r50.onnx"
settings.LIVENESS_ENABLED       # bool
```

`settings` — синглтон через `@lru_cache` (`get_settings()`), импортируется во всех сервисах/пайплайнах.

---

## 3. Итог

- Структура проекта — корректная.
- Настройки централизованы: **схема/дефолты — `app/core/config.py` (`Settings`)**, **значения — `.env`** (плюс переменные окружения и docker `environment:` с приоритетом). Доступ — `from app.core.config import settings`.

*Отчёт тестовый — сгенерирован для проверки папки `reports/`.*
