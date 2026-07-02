# Deploy Runbook — FaceID Core

Операционный runbook развёртывания FaceID Core. CPU- и GPU-пути, healthcheck,
smoke-тесты, откат, секреты/сертификаты, заметки по 152-ФЗ, масштабирование и
coverage-gate.

> Аппаратный контекст: локальная dev-машина без NVIDIA CUDA (Radeon iGPU) —
> только CPU-сборка. Production-хосты могут иметь GPU; архитектура под CPU не
> сужается (см. memory `hw-no-cuda-gpu`, `dev-vs-prod-hardware`).

---

## 1. Предварительные требования

### Общие
- Docker Engine ≥ 24.0 + Docker Compose v2 (`docker compose`).
- Портов: 8080 (LB HTTP), 8443 (LB HTTPS), 8000 (api, внутренний), 5432 (pg),
  6379 (redis), 9000/9001 (minio).
- Сетевой доступ хостов друг к другу (для two-node-см. `docs/two-node-stand.md`).

### Секреты (НЕ коммитить)
- `certs/` — TLS-сертификаты для `api_lb` (nginx, `/etc/nginx/ssl`). Файлы
  **`cert.pem`** + **`key.pem`** (имена фиксированы в `infrastructure/nginx/api_lb.conf`).
  На dev — self-signed: `bash infrastructure/nginx/generate_self_signed.sh` из корня
  репозитория. На Windows (Git Bash) — `MSYS_NO_PATHCONV=1 bash ...` (иначе `-subj
  "/CN=localhost"` превращается в Windows-путь и сертификат не генерируется).
- `DATABASE_URL`, `REDIS_URL`, `MINIO_ACCESS_KEY/SECRET_KEY`, JWT-секрет —
  через `.env` рядом с compose (в `.gitignore`) или переменные окружения оркестратора.
- Ключ AES-256 для шифрования эмбеддингов — через `ENCRYPTION_KEY` (см.
  `app/core/config.py`); ротация — отдельная процедура (не в этом runbook).

> **`.env` vs docker-compose**: корневой `.env` (если есть) нацелен на **локальный
> host-запуск** Python (`REDIS_HOST=localhost`, `DATABASE_URL=...@localhost:5432`).
> Docker Compose автоматически интерполирует `.env`, что ломает docker-сервисы
> (`localhost` в контейнере = сам контейнер, redis/postgres недоступны). Поэтому
> в `docker-compose.yml` для `worker`/`worker_metrics` host-конфликтующие
> переменные прописаны явно (`REDIS_HOST=redis`,
> `DATABASE_URL=postgresql://...@postgres:5432/faceid`) — без `${...:-...}`
> интерполяции. `.env` используется только для host-run; не клади туда docker-
> service-name-конфликты.

### Для GPU-пути (production)
- NVIDIA driver + **nvidia-container-toolkit**.
- Проверка доступности GPU в контейнере:
  ```bash
  docker run --rm --gpus all nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 nvidia-smi
  ```
- onnxruntime-gpu==1.18.0 собран под CUDA 11.8 + cuDNN 8 — базовый образ
  `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` (см. `infrastructure/docker/*.gpu.Dockerfile`).

---

## 2. Сборка: CPU (dev / production без GPU)

Базовые Dockerfile'ы (`infrastructure/docker/api.Dockerfile`, `worker.Dockerfile`)
на `python:3.11-slim` + `onnxruntime==1.18.0` (CPU). Энкодер работает на
`CPUExecutionProvider`; `ONNX_ARCFACE_PROVIDERS=auto` мягкоfallback'ит на CPU.

```bash
docker compose build
docker compose up -d
```

По умолчанию поднимаются: `postgres`, `postgres_test`, `redis`, `minio`, `api`,
`api_lb`, `worker`, `prometheus`. Профильные/`disabled`-сервисы
(`fast_worker`, `worker_metrics`, `worker_fast/heavy`, `autoscaler`) — через
`--profile <name>`.

## 3. Сборка: GPU (production с CUDA)

GPU-override переключает `api`/`worker` на CUDA-образы, резервирует GPU и
ставит `ONNX_ARCFACE_PROVIDERS=cuda`:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

GPU-override (`docker-compose.gpu.yml`) делает:
- `build.dockerfile` → `*.gpu.Dockerfile` (nvidia/cuda 11.8 + cuDNN 8 + python3.11);
- `deploy.resources.reservations.devices` — GPU от nvidia driver;
- `ONNX_ARCFACE_PROVIDERS=cuda` → `CUDAExecutionProvider` (энкодер + SCRFD в
  `app/ml/runtime.py`); fallback на CPU сохранён в коде;
- увеличенный `EMBED_BATCH_SIZE` (32/16) — на GPU батч окупается.

> onnxruntime-gpu и onnxruntime — конфликтующие пакеты; не смешивайте
> `requirements.txt` и `requirements-gpu.txt` в одном образе. Версии держать
> синхронно (оба 1.18.0).

### DirectML (Windows, AMD/Intel iGPU) — альтернатива
Если на Windows-хосте есть DirectX-ускорение, можно собрать CPU-образ, но
поставить `onnxruntime-directml` вручную и задать `ONNX_ARCFACE_PROVIDERS=directml`.
Это не основной путь; для production рекомендован CUDA.

---

## 4. Bring-up и healthcheck

После `up -d` ждать выхода healthcheck'ов в `healthy`:

```bash
docker compose ps          # столбец STATUS — (healthy)
docker compose logs -f api worker
```

Healthcheck'и (определены в `docker-compose.yml`):
| Сервис      | Проверка                                                     |
|-------------|--------------------------------------------------------------|
| postgres    | `pg_isready -U postgres`                                     |
| redis       | `redis-cli ping`                                             |
| minio       | `curl /minio/health/ready`                                   |
| api         | `urllib.urlopen('http://127.0.0.1:8000/ready')`              |
| api_lb      | `wget http://localhost/healthz` (nginx отдаёт 200 напрямую)  |
| worker      | `redis.from_url(REDIS_URL).ping()` (брокер достижим)         |

`restart: unless-stopped` на всех长期-сервисах — авторестарт после падения/перезагрузки хоста.

Готовность API напрямую (минуя LB):
```bash
curl -fsS http://localhost:8000/ready      # 200 OK
curl -fsS http://localhost:8000/health     # 200 OK
```

---

## 5. Smoke-тесты

Через LB (порт 8080 HTTP / 8443 HTTPS), с JWT:

```bash
TOKEN=<ваш-jwt>
# Liveness (passive) — бинарные real/spoof + spoofing_indicators
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"image":"<base64-jpeg>"}' http://localhost:8080/api/v1/liveness

# Upload эталона
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"user_id":"ext_smoke_1","image":"<base64-jpeg>"}' \
  http://localhost:8080/api/v1/upload

# Verify
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"user_id":"ext_smoke_1","image":"<base64-jpeg>","require_liveness":true}' \
  http://localhost:8080/api/v1/verify
```

Ожидаемые поля ответа `/verify`: `status` (`match`/`low_confidence`/`no_match`/
`spoof_detected`/`quality_reject`/`retry`/...), `match_score`, `similarity`
(legacy-алиас), `confidence` (`high`/`medium`/`low`/`null`),
`liveness_passed`, `spoofing_indicators` (`{real_prob, spoof_prob}`).

При окклюзии (маска/очки) — `status="retry"`, `reason="remove_occlusion"`: клиент
просит снять окклюзию и пере-вызывает `/verify`.

### Active liveness gate (high-security admission)

Passive-модель MiniFASNetV2 **не различает cutout-атаку** (фото с вырезом /
маска из распечатки) — cutout классифицируется как `real` P=0.976. Для допуска
на объект (high-security) допуск по passive небезопасен. Решение — **active
challenge как обязательный gate**: при `require_liveness=true` и
`LIVENESS_ACTIVE_REQUIRED=true` допуск выдаётся **только** через active proof
(`liveness_token` из WS-challenge). Cutout/print — статичны, действия не
выполняют → `is_live=False` → токен не выдаётся → `/verify` без токена → 403.

Env (high-security deploy):
```
LIVENESS_ENABLED=true
LIVENESS_ACTIVE_ENABLED=true
LIVENESS_ACTIVE_REQUIRED=true
```

Smoke-flow active-допуска:
```bash
# 1) init — получить challenge_id + ws_token + actions
curl -s -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/api/v1/liveness/challenge/init
# 2) WS /api/v1/liveness/challenge/stream?challenge_id=...&ws_token=... —
#    клиент стримит кадры, выполняя actions (turn/nod/blink) → result + liveness_token
# 3) verify с active proof
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"user_id":"ext_smoke_1","image":"<base64-jpeg>","require_liveness":true,
       "liveness_mode":"active","liveness_token":"<token>"}' \
  http://localhost:8080/api/v1/verify_base64
# → liveness_passed=true (active proof); passive-скор не влияет
```

При `require_liveness=true` **без** active proof (passive) → `403
active_liveness_required` с инструкцией пройти challenge.

**Носители active proof**: только `/verify_base64` (sync fast-path, онлайн-допуск)
принимает `liveness_mode=active` + `liveness_token`. `/verify` (multipart),
`/verify_async_file` и `/verify_async` не несут токен — при
`LIVENESS_ACTIVE_REQUIRED=true` и `require_liveness=true` они возвращают 403 с
направлением на `/verify_base64`; для access-control их использовать не нужно.

`LIVENESS_ACTIVE_REQUIRED` default `false` (backward-compat; passive-допуск
разрешён). High-security deploys обязаны ставить `true`.

Проверка контракта API: `python -m pytest tests/unit/test_api_contract.py -q`.

---

## 6. Откат

- **Откат версии образа**: пересобрать с прошлым коммитом
  `git checkout <prev> && docker compose build && docker compose up -d`.
- **Откат GPU→CPU**: убрать `-f docker-compose.gpu.yml`, пересобрать базовые образы.
- **Откат БД-схемы**: Alembic (`alembic downgrade -1`); эмбеддинги не трогать
  без ротации ключа (AES-шифр привязан к `ENCRYPTION_KEY`).
- Состояние `postgres_data` / `minio_data` — named volumes; `docker compose down -v`
  их **удаляет** — применять только для чистого dev-окружения.

---

## 7. Секреты, сертификаты, 152-ФЗ

- **Эмбеддинги шифруются AES-256** при записи (`app/...`); `ENCRYPTION_KEY` —
  секрет оркестратора, не в образе/репозитории.
- **Исходные фото не хранятся**: после извлечения эмбеддинга байт-картинка
  удаляется из памяти; в MinIO биометрия не складируется.
- **Логи без биометрии**: `app/core/logger.py` → `BiometryRedactionFilter`
  вычищает base64-блобы, ndarray-эмбеддинги, биометрические ключи. Не логировать
  `image`/`embedding`/`crop` вручную.
- **HTTPS**: продакшн-трафик — только через `api_lb` на :8443 (TLS). :8080 —
  только внутренний/healthz.
- **JWT auth** на всех бизнес-эндпоинтах под `/api/v1`.

---

## 8. Масштабирование

- **Worker**: `docker compose up --scale worker=<N>` (горизонтально; каждый
  контейнер — свой пул). CPU-limits `cpus: "2.0"` — снимать/поднимать под хост.
- **API**: за `api_lb` (nginx upstream); scale `api` + обновить upstream.
- **GPU-путь**: `EMBED_BATCH_SIZE` поднять (override уже ставит 32/16);
  `ONNX_INTRA_OP_THREADS=0` — onnxruntime сам выбирает под GPU.
- **Кеш/очередь**: Redis уже в стеке; TTL-кеш и Celery-брокер на нём.
- **Автоскейлинг** — `autoscaler`-сервис (профиль `disabled`), см.
  `docs/two-node-stand.md`.

---

## 9. Coverage-gate (CI)

Измерение покрытия (без gate, не падает на частичных прогонах):
```bash
python -m pytest --cov=app --cov-branch --cov-report=term-missing --cov-report=html
```

Gate (CI, полный suite unit+integration) — KPI CLAUDE.md ≥90%:
```bash
python -m pytest --cov=app --cov-branch --cov-fail-under=90
```

`fail_under` **не зашит** в `pyproject.toml` намеренно: чтобы
`pytest --cov` на частичном прогоне (только `-m unit`, ~39%) не падал. Gate
задаётся явным `--cov-fail-under` в CI. Текущее покрытие unit-only ≈39%,
полного suite — выше; цель — ratchet к 90% на полном наборе.

Конфиг coverage: `[tool.coverage.run]` (source=`app`, branch=true, omit
migrations/tests) и `[tool.coverage.report]` (exclude_lines) — в `pyproject.toml`.

---

## 10. Диагностика провайдера энкодера

После старта проверить, какой ONNX-провайдер реально поднялся:
```bash
docker compose logs api worker | grep -i "execution_provider\|cuda\|onnx"
```
Или in-process:
```python
import onnxruntime as ort
print(ort.get_available_providers())
```
На CPU-хосте ожидается `CPUExecutionProvider`; на GPU — `CUDAExecutionProvider`
(+`CPUExecutionProvider` как fallback). Если на GPU-хосте виден только CPU —
проверить `nvidia-container-toolkit` и `deploy.resources.reservations.devices`.