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