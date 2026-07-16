"""Locust-нагрузочный скрипт для FaceID Core /verify.

Два режима (теги):
  - @tag("enqueue"): POST /api/v1/verify_base64 → сразу pending {job_id}.
    Нагрузка на API admission/guard + queue-delay. RPS приёма, НЕ ML-throughput.
  - @tag("e2e"): POST /api/v1/verify_base64 → поллинг
    GET /api/v1/jobs/{job_id}/wait?timeout=2000 до терминала.
    Реальный end-to-end SLO (p95 полной latency). По умолчанию.

Запуск (100 RPS, 2 минуты):

    # 1. auth: проще всего отключить на dev (как tests/conftest.py:23):
    AUTH_ENABLED=false docker compose up -d --build api worker postgres redis minio

    # 2. locust (PYTHONUTF8=1 обязательно на Windows — pyproject.toml содержит
    #    кириллицу, locust парсит его через charmap и падает без UTF-8):
    PYTHONUTF8=1 locust -f scripts/load_test_locust.py \
        --host http://localhost:8000 \
        --headless -u 100 -r 100 -t 120s \
        --tags e2e \
        --html=benchmarks/locust_e2e_100rps.html

    # только enqueue-нагрузка (RPS приёма):
    locust -f scripts/load_test_locust.py --host http://localhost:8000 \
        --headless -u 100 -r 100 -t 60s --tags enqueue

Примечание: проект исторически использует k6 (benchmarks/, scripts/capture_benchmark.ps1).
Locust добавлен как Python-native для dev-команды; контракты 1:1 с load_test_verify_base64.js.

Auth (если AUTH_ENABLED=true): передать через env API_KEY (settings.API_KEYS) —
скрипт подставит X-API-Key. Либо AUTH_ENABLED=false на dev.

152-ФЗ: скрипт шлёт тестовое фото tests/data/person1_small.jpg (артефакт репо,
НЕ биометрия реальных пользователей). base64 не пишется в лог.
"""
from __future__ import annotations

import base64
import os
import uuid
from pathlib import Path

from locust import HttpUser, between, constant_pacing, events, task, tag

# --- Конфигурация через env ---------------------------------------------------
BASE64_FILE = os.getenv(
    "LOADTEST_B64_FILE",
    str(Path(__file__).resolve().parent.parent / "tests" / "data" / "person1_small.b64.txt"),
)
JPEG_FILE = os.getenv(
    "LOADTEST_JPEG_FILE",
    str(Path(__file__).resolve().parent.parent / "tests" / "data" / "person1_small.jpg"),
)
USER_ID = os.getenv("LOADTEST_USER_ID", "1")
REQUIRE_LIVENESS = os.getenv("LOADTEST_REQUIRE_LIVENESS", "false").lower() == "true"
WAIT_TIMEOUT_MS = int(os.getenv("LOADTEST_WAIT_TIMEOUT_MS", "2000"))
POLL_INTERVAL = float(os.getenv("LOADTEST_POLL_INTERVAL", "0.05"))
# API key (если AUTH_ENABLED=true и settings.API_KEYS задан)
API_KEY = os.getenv("API_KEY", "")

# --- Base64 образ (один раз на старте, не на каждый запрос) ------------------
_IMAGE_B64: str = ""


def _load_image_b64() -> str:
    """Загрузить base64 тестового фото: из .b64.txt если есть, иначе кодировать jpg."""
    global _IMAGE_B64
    if _IMAGE_B64:
        return _IMAGE_B64
    b64_path = Path(BASE64_FILE)
    if b64_path.exists():
        _IMAGE_B64 = b64_path.read_text(encoding="utf-8").strip()
        return _IMAGE_B64
    jpg_path = Path(JPEG_FILE)
    if not jpg_path.exists():
        raise FileNotFoundError(
            f"Не найден тестовый образ: ни {b64_path}, ни {jpg_path}. "
            "Положите tests/data/person1_small.jpg или укажите LOADTEST_B64_FILE/LOADTEST_JPEG_FILE."
        )
    _IMAGE_B64 = base64.b64encode(jpg_path.read_bytes()).decode("ascii")
    return _IMAGE_B64


@events.test_start.add_listener
def _on_test_start(environment, **kwargs):
    """Предзагрузка base64 на старте + sanity-печать."""
    try:
        b64 = _load_image_b64()
        print(f"[loadtest] образ загружен: {len(b64)} символов base64; user_id={USER_ID}")
        print(f"[loadtest] require_liveness={REQUIRE_LIVENESS}; wait_timeout_ms={WAIT_TIMEOUT_MS}")
        print(f"[loadtest] API_KEY: {'задан' if API_KEY else 'нет (нужен AUTH_ENABLED=false)'}")
    except FileNotFoundError as exc:
        print(f"[loadtest] FATAL: {exc}")
        environment.runner.quit()


class VerifyUser(HttpUser):
    """Нагрузочный пользователь: бьёт по /verify с задержкой 1с (≈1 RPS на user).

    wait_time=constant_pacing(1) → ровно 1 итерация/сек на user независимо от
    latency ответа. -u 100 -r 100 → 100 RPS (если backend держит). Если backend
    не держит, pacing растягивается → реальный RPS ниже (видно в stats).
    """

    wait_time = constant_pacing(1.0)

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "X-Request-Id": str(uuid.uuid4())}
        if API_KEY:
            h["X-API-Key"] = API_KEY
        return h

    def _enqueue(self) -> str | None:
        """POST /api/v1/verify_base64 → {job_id, status}. Возвращает job_id или None."""
        payload = {
            "user_id": USER_ID,
            "image": _IMAGE_B64,
            "require_liveness": REQUIRE_LIVENESS,
        }
        with self.client.post(
            "/api/v1/verify_base64",
            json=payload,
            headers=self._headers(),
            name="POST /verify_base64 (enqueue)",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"enqueue HTTP {resp.status_code}: {resp.text[:200]}")
                return None
            try:
                body = resp.json()
            except ValueError:
                resp.failure(f"enqueue non-JSON: {resp.text[:200]}")
                return None
            job_id = body.get("job_id")
            status = body.get("status")
            # Smart-union: terminal (match/no_match/...) или pending — оба 200.
            if status in ("match", "no_match", "spoof_detected", "quality_reject"):
                # Синхронный fast-path (если включён) — терминал сразу.
                resp.success()
                return None  # поллинг не нужен
            if status in ("pending", "queued") and job_id:
                resp.success()
                return job_id
            resp.failure(f"enqueue unexpected status={status!r} job_id={job_id!r}")
            return None

    def _poll_terminal(self, job_id: str, name: str = "GET /jobs/{id}/wait (poll)") -> None:
        """Long-poll /api/v1/jobs/{job_id}/wait до терминала. timeout=2000ms.

        Сервер держит соединение до терминала или WAIT_TIMEOUT_MS. Если timeout
        истёк без терминала — повторяем (worker занят). Помечаем success только
        при терминальном status.
        """
        import time

        deadline = time.monotonic() + 30.0  # safety-cap 30с на один job
        url = f"/api/v1/jobs/{job_id}/wait?timeout={WAIT_TIMEOUT_MS}"
        while time.monotonic() < deadline:
            with self.client.get(
                url,
                headers=self._headers(),
                name=name,
                catch_response=True,
            ) as resp:
                if resp.status_code == 404:
                    resp.failure(f"job {job_id} not found")
                    return
                if resp.status_code != 200:
                    resp.failure(f"poll HTTP {resp.status_code}: {resp.text[:200]}")
                    return
                try:
                    body = resp.json()
                except ValueError:
                    resp.failure(f"poll non-JSON: {resp.text[:200]}")
                    return
                status = body.get("status")
                if status in ("match", "no_match", "spoof_detected",
                              "quality_reject", "processing_failed"):
                    resp.success()
                    return
                if status in ("done", "terminal"):
                    resp.success()
                    return
                # pending/running/timeout → сервер отдал управление, повторяем
                resp.success()  # сам wait-вызов корректен, даже если не терминал
            time.sleep(POLL_INTERVAL)
        # deadline истёк
        self.client.get(
            f"/api/v1/jobs/{job_id}",
            headers=self._headers(),
            name="GET /jobs/{id} (final-check)",
            catch_response=True,
        ).failure(f"job {job_id} не терминал за 30с")

    @task
    @tag("e2e")
    def verify_e2e(self) -> None:
        """End-to-end: enqueue + long-poll терминала. Реальный SLO p95<500ms."""
        job_id = self._enqueue()
        if job_id is None:
            return  # терминал сразу (fast-path) или ошибка enqueue
        self._poll_terminal(job_id, name="GET /jobs/{id}/wait (e2e)")

    @task
    @tag("enqueue")
    def verify_enqueue_only(self) -> None:
        """Только admission: enqueue без поллинга. RPS приёма / queue backpressure."""
        self._enqueue()