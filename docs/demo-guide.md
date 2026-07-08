# Демо-GUI FaceID Core — запуск и презентация

Веб-интерфейс для демонстрации работоспособности FaceID Core на локальном ПК.
Изображение — с веб-камеры браузера; взаимодействие — через REST/WebSocket API
сервиса. Назначение — **наглядная презентация** (upload эталона, verify, passive
и active liveness, настройка порогов), НЕ production.

> Аппаратный контекст: локальная dev-машина без NVIDIA CUDA — энкодер работает на
> CPU (ArcFace fallback). Для демо этого достаточно; latency выше production, но
> корректность сохранена (см. memory `hw-no-cuda-gpu`).

---

## 1. Что это и что НЕ production

- `AUTH_ENABLED=false`, passive liveness и active challenge включены через
  `docker-compose.demo.yml` — **только для демо**. В production auth+HTTPS обязательны.
- Кадры с камеры **не сохраняются и не логируются** (152-ФЗ): `canvas` → `fetch` →
  обнуление ссылки. Без `localStorage`/`sessionStorage` для биометрии.
- Демо-страница отдаётся FastAPI как статика на `/demo` (same-origin с `/api/v1`)
  на `http://localhost:8000`. CORS не нужен.

---

## 2. Запуск

```bash
# 1. Поднять стек в демо-режиме (auth выкл, liveness вкл):
docker compose -f docker-compose.yml -f docker-compose.demo.yml up -d --build

# 2. Дождаться healthy:
docker compose ps                  # все сервисы (healthy)

# 3. Проверить API:
curl http://localhost:8000/health  # {"status":"ok"}
```

Открыть в браузере **`http://localhost:8000/demo/``** (Chrome/Firefox).

> `getUserMedia` (доступ к камере) требует HTTPS **или** `localhost`. На
> `localhost` HTTPS не нужен — поэтому демо работает на plain HTTP `:8000`.
> На не-localhost демо-хосте нужен HTTPS (через `api_lb` `:8443`) — но тогда
> self-signed сертификат придётся принять вручную.

Браузер спросит разрешение на камеру — разрешаем.

---

## 3. Режимы (табы)

Слева — веб-камера (кнопка «Старт камеры» → «Захватить кадр»). Сверху — общие
параметры: `user_id`, `require_liveness`, `liveness_mode` (passive/active).

### Upload (запись эталона)
1. Старт камеры → «Захватить кадр».
2. Таб **Upload** → «Загрузить эталон».
3. Ответ: `embedding_id` (эталон записан в БД, шифруется AES-256).

### Verify (passive)
1. Убедитесь что `user_id` совпадает с загруженным эталоном.
2. Таб **Verify** → «Verify».
3. Ответ: `status` (match/no_match/...), `match_score`, `confidence`,
   `liveness_passed`, `spoofing_indicators` (real_prob/spoof_prob).
4. Свежий кадр берётся с камеры автоматически.

### Liveness (passive, standalone)
- Таб **Liveness** → «Проверить liveness».
- Ответ: `liveness` (true/false), `score`, `face_detected`, `spoofing_indicators`.
- Проверяет только живость кадра, без верификации личности.

### Active Challenge (защита от cutout/print)
1. Таб **Active Challenge** → «Начать challenge».
2. Сервер возвращает список действий (blink / turn_left / turn_right / nod / smile).
3. Кадры стримятся в WebSocket автоматически (~каждые 600мс) — выполните действия
   перед камерой.
4. По выполнении → «Готово (done)».
5. Ответ: `is_live`, `liveness_token` (если пройден). Статичное фото/распечатка
   действия не выполняют → `is_live=false` → токен не выдан.
6. «Verify active» → `/verify_base64` с `liveness_mode=active` + токен → допуск.

WS-контракт `/liveness/challenge/stream`: сервер шлёт `{type:"challenge", actions, deadline_ms}`
→ клиент стримит бинарные JPEG-кадры + `{cmd:"done"}`/`{cmd:"cancel"}`; ответ
`{type:"result", is_live, liveness_token, spoofing_indicators}` или
`{type:"cancelled"}`. Коды закрытия: `4401` неверный ws_token, `4409` конфликт
(уже стримит), `4410` challenge истёк/неизвестен, `4400` некорректное состояние
challenge, `4503` liveness отключён/сервер занят, `1006` разрыв.

### Config
- «Обновить» → `GET /api/v1/config` отдаёт текущие пороги (read-only):
  `FACE_MATCH_THRESHOLD`, `LIVENESS_THRESHOLD`, `LIVENESS_ENABLED`,
  `LIVENESS_ACTIVE_ENABLED`, `LIVENESS_ACTIVE_REQUIRED`, `QUALITY_GATE_MODE`.
- **Слайдер** «Client-side порог match» — влияет только на цвет бейджа в
  результатах verify (client-side интерпретация `match_score`). Серверный порог
  НЕ меняется (runtime-mutation отсутствует; меняется через env + restart).

---

## 4. Интерпретация результатов

| `status` | Значение | UI |
|---|---|---|
| `match` | `match_score ≥ FACE_MATCH_THRESHOLD` (0.6) | зелёный |
| `no_match` | `match_score ≤ LOW_THRESHOLD` (0.3) | красный |
| `low_confidence` | серая зона 0.3–0.6 | серый, рекомендуется active challenge |
| `spoof_detected` | passive liveness не прошёл | красный, бары real/spoof |
| `quality_reject` | не прошёл quality gate (блюр/яркость/контраст) | жёлтый + reason |
| `retry` | окклюзия (маска/очки), `reason=remove_occlusion` | оранжевый, снять и повторить |
| `no_face` | лицо не обнаружено | жёлтый |
| `processing_failed` | внутренняя ошибка, `error_code` | красный |

`confidence`: `high` (≥0.6), `medium` (0.3–0.6), `low` (<0.3), `null` (не-match).
`challenge_recommended=true` → серая зона, стоит пройти active challenge.

---

## 5. Презентация anti-spoof (cutout-атака)

Ключевая ценность системы — защита от спуфинга. Демонстрация:

1. **Загрузите свой эталон** (Upload) — лицо в кадре.
2. **Passive verify с живого лица** → `match`, `liveness_passed=true`. Базовый сценарий.
3. **Cutout/print-атака на passive**: покажите перед камерой фото лица на экране
   телефона или распечатку. Passive MiniFASNet **обманывается** cutout
   (`real_prob≈0.976` — модель принимает за живое). Verify может дать ложный
   `match` с `liveness_passed=true`. Это демонстрирует **уязвимость passive**.
4. **Active challenge на том же cutout**: фото статично — не моргает, не поворачивает
   голову, не кивает → `is_live=false` → `liveness_token` **не выдан** →
   «Verify active» недоступен / отклонён. **Допуск не выдан.**
5. Вывод: active-протокол закрывает класс physical-spoof (cutout/print/replay),
   который ложит passive-модель. Поэтому high-security deploys ставят
   `LIVENESS_ACTIVE_REQUIRED=true` (допуск только через active proof).

> В демо-override `LIVENESS_ACTIVE_REQUIRED` оставлен `false` (default), чтобы
> можно было показать уязвимость passive. Для демонстрации hard-gate поставьте
> `LIVENESS_ACTIVE_REQUIRED=true` в `docker-compose.demo.yml` и пересоберите — тогда
> passive-verify с `require_liveness=true` вернёт `403 active_liveness_required`.

---

## 6. Архитектура демо

```
Браузер (http://localhost:8000/demo/)
  │  getUserMedia → canvas → base64 / Blob
  ├─ POST /api/v1/upload_base64      (Upload эталона)
  ├─ POST /api/v1/verify_base64      (Verify passive/active)
  ├─ POST /api/v1/liveness           (multipart, passive standalone)
  ├─ POST /api/v1/liveness/challenge/init  (active)
  ├─ WS   /api/v1/liveness/challenge/stream (binary JPEG frames → token)
  └─ GET  /api/v1/config             (read-only пороги)
        │
        ▼
  FastAPI (api:8000)  ← StaticFiles mount /demo (demo/index.html, app.js, styles.css)
```

- Статика: `app/main.py` → `app.mount("/demo", StaticFiles(directory="<repo>/demo", html=True))`.
- Read-only конфиг: `app/api/routes/config.py` → `GET /api/v1/config` (без секретов).
- Docker: `infrastructure/docker/api.Dockerfile` → `COPY demo ./demo`.

---

## 7. Безопасность / заметки

- **Production**: убрать `StaticFiles` mount `/demo` (или защитить auth+HTTPS);
  `AUTH_ENABLED` обязательно `true`; `docker-compose.demo.yml` не использовать.
- **WebSocket** в демо — `ws://` (не `wss://`), работает только на localhost.
  На не-localhost нужен `wss://` через `api_lb` `:8443`.
- **Кадры** не пишутся на диск нигде (браузер + сервер: исходные фото удаляются
  после извлечения эмбеддинга, эмбеддинги шифруются AES-256 — см. README).
- Если `GET /api/v1/config` возвращает `LIVENESS_ENABLED=false` — проверьте, что
  запуск через `docker-compose.demo.yml` (базовый compose держит `false`).
- Если `/demo/` отдаёт 404 — проверьте `docker compose exec api ls /app/demo`
  (Dockerfile должен скопировать `demo/`).

---

## 8. Desktop-демо (Windows, нативное окно)

Альтернатива веб-демо — **нативное desktop-приложение** на tkinter + OpenCV
(`demo/desktop_demo.py`), запускаемое двойным кликом по `demo/run_demo.bat`.
Назначение то же — презентация сервиса, но без браузера/HTTPS/getUserMedia:
камера через `cv2.VideoCapture(0)`, API через `requests`/`websocket-client`,
превью кадра через `PIL.ImageTk`.

### Когда выбирать desktop, а не web

- Презентация на Windows-ПК без браузера или где `getUserMedia` неудобен
  (корпоративные политики, self-signed HTTPS).
- Нужен «one-click» запуск: .bat сам поднимает demo-стек и открывает окно.
- Веб-демо (`/demo/`) остаётся доступным параллельно — desktop его не заменяет.

### Запуск

```
Двойной клик demo/run_demo.bat
```

`.bat` последовательно проверяет: Python ≥3.10 в PATH → наличие пакетов
(`cv2, requests, websocket, PIL`; при отсутствии ставит из
`demo/requirements-demo.txt`) → Docker + запущенный демон (`docker info`) →
запускает `python demo/desktop_demo.py`. Консоль .bat остаётся открытой — туда
пишется stdout/stderr Python (traceback при краше → `pause`).

В окне:

1. **«Запустить сервис»** — приложение само выполняет
   `docker compose -f docker-compose.yml -f docker-compose.demo.yml up -d --build
   postgres redis minio api worker` (отсекает `api_lb`/`prometheus`/профильные
   worker'ы — прод-сервисы демо не нужны) и ждёт `GET /ready` (≤90с).
2. Превью камеры живёт сразу (камера стартует до сервиса — превью не зависит от API).
3. **Эталон**: «Снять с камеры» или «Из файла…» → `POST /upload_base64` →
   `embedding_id`.
4. **«Верифицировать (камера)»** — сама снимает кадр → `POST /verify_base64`.
   В demo-override `USE_FAST_PATH=false` — запрос идёт через async-очередь
   `face_verify_queue` (тот же worker, что и новые async-роуты), ответ `pending` →
   long-poll `/jobs/{id}/wait?timeout=2000` до терминала. Это убирает «зависание на
   pending», бывшее при отключённом Celery. Чекбокс «требовать liveness (passive)»
   по умолчанию **вкл** (нагляднее: ответ несёт `liveness_passed`/`liveness_score`).
5. **«Остановить сервис»** — `docker compose ... down -v` (чистит volumes с
   биометрией, 152-ФЗ). То же автоматически при закрытии окна.

### Минимум действий (автоматизация)

- **Одна кнопка** «Верифицировать (камера)» сама снимает кадр — отдельный «захват»
  не нужен.
- **Retry (окклюзия)**: при `status=retry` (`reason=remove_occlusion`) UI сам
  показывает оверлей «Снимите маску/очки» (по `quality_details.occlusion_flags`)
  и переименовывает кнопку в «Переснять». Без авто-зацикливания — пересъёмка
  только по клику (пользователь должен реально снять предмет).
- **Active challenge авто-переход**: при `challenge_recommended=true` или
  `confidence=low` (серая зона) и включённом чекбоксе «авто active-challenge»
  (default вкл) окно само стартует challenge — оверлей «поверните голову /
  моргните», стрим JPEG-кадров каждые 600мс (≤30) в WS. По `is_live=true` +
  `liveness_token` автоматически вызывает `/verify_base64` с
  `liveness_mode=active` + токен (single-use, TTL 120с). Пользователь только
  выполняет действия перед камерой. Кнопка «Запустить вручную» — для повтора.
- Active gate — **soft**: `LIVENESS_ACTIVE_REQUIRED` не трогается (остаётся
  `false`), `docker-compose.demo.yml` не правится. Passive-verify работает;
  active — по рекомендации или вручную.

### Безопасность (152-ФЗ), как и в веб-демо

- Кадры **только в памяти**: `cv2.imencode` → bytes/base64 в локальных
  переменных, `del`/GC после отправки. Никаких `cv2.imwrite`, temp-файлов,
  логов с base64. Лог-зона — только человекочитаемые строки (status, score).
- `liveness_token` — атрибут `ChallengeSession` в памяти, не файл/env; уходит в
  запрос немедленно и собирается GC.
- `down -v` чистит volumes postgres/minio при остановке/закрытии.
- `AUTH_ENABLED=false` — только demo-override; UI не хранит JWT/X-API-Key,
  не читает `.env` (прод-секреты не светятся). `GET /api/v1/config` — только
  6 read-only порогов.

### Подводные камни

- **Закрывайте окно кнопкой «Стоп» / крестиком окна**, не консоль .bat — иначе
  Python убивается без `down -v` (контейнеры остаются, volumes не чистятся).
  Крестик окна корректно стопает сервис в daemon-потоке.
- **Камера занята** другим приложением (вкл. веб-демо в браузере с активной
  `getUserMedia`) → `VideoCapture(0)` упадёт; лог подскажет закрыть конфликтующее
  приложение. Web-демо и desktop одновременно на одной камере не работают.
- Только Windows (`.bat`); tkinter+cv2 кроссплатформенны, но лаунчер
  Windows-специфичен. Docker Desktop должен быть запущен до старта .bat.
- `opencv-python` (не headless) — нужен `VideoCapture` к физической камере;
  вынесен в `demo/requirements-demo.txt`, отдельно от основного `requirements.txt`
  (демо-зависимости не нужны прод-сервису).
- Verify-путь (demo): `USE_FAST_PATH=false` → `verify_base64` ставит job в
  `face_verify_queue` и возвращает `pending`; приложение long-poll'ит
  `/jobs/{id}/wait?timeout=2000` до терминала (`done`/`error`/`expired`/`failed`),
  max 30с. Это основной путь в демо (не fallback): Celery-роуты в demo-override
  отключены, а `face_verify_queue` потребляется одним worker-контейнером.
- **Диагностика**: стадии лаунчера пишутся в `demo/_demo_launcher.log`
  (`=== launcher start ===` → `[1/4] Python` → `[2/4]` → `[3/4] Dependencies OK`
  → `[4/4] Docker OK` → `[launcher] starting desktop_demo.py` → `rc=…`), а
  бизнес-логика (upload/verify/challenge) — в `demo/_demo_ui.log` (только
  человекочитаемые метаданные: status/score/reason, без кадров/эмбеддингов/base64,
  152-ФЗ). При «окно закрылось мгновенно / падает сразу» достаточно прислать
  `_demo_launcher.log`; при сбое внутри приложения — `_demo_ui.log` и
  `_demo_stdout.log` (traceback desktop_demo.py). Все `*.log` в `demo/` gitignored.