# tests/unit/test_verify_task.py — покрытие celery-task verify_task.
#
# process_verify_job — точка входа celery; _process_verify_job — основная
# корутина (gate → get job → download → extract → spoof/liveness-gate →
# search → decision → save → webhook → delete). Вызываем task напрямую
# (без брокера) с замоканными зависимостями (MinIO/DB/repos/Redis/webhook).
# Покрывает: active-gate, job-not-found, already-done, spoof-branch,
# liveness-fail, happy-match — основную бизнес-логику worker'а.

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

import app.workers.tasks.verify_task as vt
from app.models.verification_job import JobStatus


# ---------------------------- фейковый Redis ----------------------------

class FakeRedis:
    def __init__(self) -> None:
        self.set_calls: list[tuple] = []
        self.values: dict[str, str] = {}

    def setex(self, key, value=None, ttl=None, **kw):
        # verify_task вызывает setex(key, value, ttl=N) — позиционно (key,value,ttl).
        # Совместимо и с (key,ttl,value) — различаем по типу второго аргумента.
        if isinstance(value, int) and ttl is not None:
            value, ttl = ttl, value
        self.set_calls.append((key, value, ttl))
        self.values[key] = value

    def set(self, key, value, ttl=None, **kw):
        self.values[key] = value


# ----------------------- фейковая async DB-сессия -----------------------
# AsyncSessionLocal() — async ctx-manager; используется несколько раз за
# корутину (get-job, spoof-log, save-log). Каждый вызов даёт «сессию», в
# которую repos прокидывают себя.

class FakeSession:
    def __init__(self) -> None:
        self.committed = 0

    async def commit(self) -> None:
        self.committed += 1


class FakeAsyncSessionLocal:
    """factory: AsyncSessionLocal() -> async ctx-manager c FakeSession."""

    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []

    def __call__(self):
        session = FakeSession()
        self.sessions.append(session)
        outer = self

        class _Ctx:
            async def __aenter__(self_inner):
                return session

            async def __aexit__(self_inner, *exc):
                return False

            def __getattr__(self_inner, name):
                # делегируем атрибуты (на случай прямого обращения) к session
                return getattr(session, name)

        return _Ctx()


# ------------------------------ фейковые repos ------------------------------

class FakeJobRepo:
    # класс-атрибуты — конфигурация сценария (job, update-calls)
    job: object = None
    updates: list[tuple] = []

    def __init__(self, db) -> None:
        self.db = db

    async def get_by_id(self, job_id: str):
        return FakeJobRepo.job

    async def update(self, job_id, status=None, error=None, result=None):
        FakeJobRepo.updates.append((job_id, status, error, result))


class FakeVerificationRepo:
    logs: list[dict] = []

    def __init__(self, db) -> None:
        self.db = db

    async def create_log(self, **kwargs):
        FakeVerificationRepo.logs.append(kwargs)


class FakeSearchService:
    top_k_result: list = []

    def __init__(self, embedding_repo) -> None:
        pass

    async def search_top_k(self, embedding, k=None):
        return FakeSearchService.top_k_result


class FakeMinioClient:
    image_bytes: bytes = b"image-bytes"
    deleted: list[str] = []
    get_raises: bool = False

    def __init__(self) -> None:
        pass

    def get_image(self, url: str) -> bytes:
        if FakeMinioClient.get_raises:
            raise RuntimeError("minio down")
        return FakeMinioClient.image_bytes

    def delete_image(self, url: str) -> None:
        FakeMinioClient.deleted.append(url)


class FakeVerificationService:
    # статические ответы сценариев
    features: dict = {}
    decision: dict = {}

    def __init__(self, embedding_repo=None, verification_repo=None, pipeline=None) -> None:
        pass

    def extract_features(self, image_bytes: bytes) -> dict:
        return FakeVerificationService.features

    def make_decision(self, embedding, top_k, liveness) -> dict:
        return FakeVerificationService.decision


def _make_job(status=JobStatus.pending, created=True) -> SimpleNamespace:
    return SimpleNamespace(
        id="job-1",
        status=status,
        created_at=datetime(2026, 7, 8, 12, 0, 0) if created else None,
    )


# ------------------------------- фикстура --------------------------------

@pytest.fixture
def task_env(monkeypatch):
    """Разводит окружение worker-task: мокает DB/repos/MinIO/Redis/webhook/
    pipeline/service/liveness/decrement. Сбрасывает счётчики между тестами."""
    fake_redis = FakeRedis()
    fake_session_local = FakeAsyncSessionLocal()

    monkeypatch.setattr(vt, "redis_client", fake_redis)
    monkeypatch.setattr(vt, "AsyncSessionLocal", fake_session_local)
    monkeypatch.setattr(vt, "MinioClient", FakeMinioClient)
    monkeypatch.setattr(vt, "VerificationJobRepository", FakeJobRepo)
    monkeypatch.setattr(vt, "VerificationRepository", FakeVerificationRepo)
    monkeypatch.setattr(vt, "SearchService", FakeSearchService)
    monkeypatch.setattr(vt, "VerificationService", FakeVerificationService)
    monkeypatch.setattr(vt, "get_pipeline", lambda: object(), raising=False)
    monkeypatch.setattr(vt, "LIVENESS_ACTIVE_REQUIRED", False, raising=False)
    monkeypatch.setattr(vt, "settings", SimpleNamespace(
        LIVENESS_ACTIVE_REQUIRED=False, FAISS_ENABLED=False), raising=False)

    async def _no_webhook(*a, **kw):
        return None

    monkeypatch.setattr(vt, "_webhook_notify_direct", _no_webhook)

    def _no_decrement():
        return None

    monkeypatch.setattr(vt, "decrement_active", _no_decrement)

    class _Liveness:
        @staticmethod
        def is_passed(signals):
            return False

        @staticmethod
        def fuse(signals):
            return {"score": 0.2, "risk": "spoof"}

    monkeypatch.setattr(vt, "LivenessService", _Liveness)

    # сброс счётчиков
    FakeJobRepo.updates = []
    FakeVerificationRepo.logs = []
    FakeSearchService.top_k_result = []
    FakeMinioClient.deleted = []
    FakeMinioClient.get_raises = False
    FakeVerificationService.features = {}
    FakeVerificationService.decision = {}

    return SimpleNamespace(
        redis=fake_redis,
        session_local=fake_session_local,
    )


# ------------------------------- сценарии --------------------------------

def _call_task(**kwargs):
    """Вызывает bind=True celery-task синхронно: push_request даёт request
    context (self.request.retries=0), .run — оригинальная функция с bound self.
    Без брокера/backend."""
    task = vt.process_verify_job
    task.push_request()
    try:
        return task.run(**kwargs)
    finally:
        task.pop_request()


@pytest.mark.unit
def test_active_liveness_required_gate(task_env, monkeypatch):
    """require_liveness=True + LIVENESS_ACTIVE_REQUIRED=True → job marked
    failed (defense-in-depth gate, без retry). Покрывает gate + _mark_verify_job_failed."""
    monkeypatch.setattr(vt, "LIVENESS_ACTIVE_REQUIRED", True, raising=False)
    vt.settings.LIVENESS_ACTIVE_REQUIRED = True
    FakeJobRepo.job = _make_job()

    result = _call_task(
        job_id="job-1",
        image_url="img-1",
        user_id="42",
        require_liveness=True,
    )
    assert result is None
    # job помечен failed
    assert any(u[1] == JobStatus.failed for u in FakeJobRepo.updates)
    # результат закеширован в redis (job:job-1)
    assert "job:job-1" in task_env.redis.values


@pytest.mark.unit
def test_job_not_found(task_env):
    """get_by_id → None → LookupError → тихий возврат (без retry)."""
    FakeJobRepo.job = None
    result = _call_task(job_id="missing", image_url="img-1", user_id="42")
    assert result is None


@pytest.mark.unit
def test_job_already_done_skipped(task_env):
    """job.status=done → return без повторной обработки (идемпотентность)."""
    FakeJobRepo.job = _make_job(status=JobStatus.done)
    result = _call_task(job_id="job-1", image_url="img-1", user_id="42")
    assert result is None
    # не было update на processing/done, не было логов
    assert FakeJobRepo.updates == []


@pytest.mark.unit
def test_spoof_branch(task_env):
    """features.status='spoof' → spoof-результат сохраняется (отдельная ветка,
    без search/decision). Покрывает строки 297-343."""
    FakeJobRepo.job = _make_job()
    FakeVerificationService.features = {
        "status": "spoof",
        "liveness": {"score": 0.12, "risk": "spoof"},
        "timings": {"preprocess_ms": 1.0, "detect_ms": 2.0, "encode_ms": 3.0,
                    "liveness_ms": 4.0, "total_pipeline_ms": 10.0},
        "bbox_source": "fast",
        "embedding": [0.0] * 512,
    }
    _call_task(job_id="job-1", image_url="img-1", user_id="42")
    # job переведён processing → done со spoof-результатом
    statuses = [u[1] for u in FakeJobRepo.updates]
    assert JobStatus.processing in statuses
    assert JobStatus.done in statuses
    done_update = next(u for u in FakeJobRepo.updates if u[1] == JobStatus.done)
    assert done_update[3]["status"] == "spoof"
    assert FakeMinioClient.deleted == ["img-1"]


@pytest.mark.unit
def test_liveness_gate_fail(task_env, monkeypatch):
    """require_liveness=True + liveness_passed=False (но features не spoof) →
    spoof-результат (gate до search). Покрывает строки 357-371."""
    FakeJobRepo.job = _make_job()
    FakeVerificationService.features = {
        "status": "ok",
        "liveness": {"score": 0.3, "risk": "spoof"},
        "timings": {"preprocess_ms": 1.0, "detect_ms": 2.0, "encode_ms": 3.0,
                    "total_pipeline_ms": 10.0},
        "bbox_source": "fast",
        "embedding": [0.0] * 512,
    }
    # LivenessService.is_passed → False (уже в фикстуре)
    _call_task(job_id="job-1", image_url="img-1", user_id="42",
               require_liveness=True)
    done_update = next(u for u in FakeJobRepo.updates if u[1] == JobStatus.done)
    assert done_update[3]["status"] == "spoof"
    assert done_update[3]["liveness_passed"] is False


@pytest.mark.unit
def test_happy_match(task_env, monkeypatch):
    """liveness_passed=True → search + decision(match) → done. Основной путь."""
    # is_passed → True
    class _LivePass:
        @staticmethod
        def is_passed(signals):
            return True

        @staticmethod
        def fuse(signals):
            return {"score": 0.95, "risk": "live"}

    monkeypatch.setattr(vt, "LivenessService", _LivePass)

    FakeJobRepo.job = _make_job()
    FakeVerificationService.features = {
        "status": "ok",
        "liveness": {"score": 0.95, "risk": "live"},
        "timings": {"preprocess_ms": 1.0, "detect_ms": 2.0, "encode_ms": 3.0,
                    "liveness_ms": 4.0, "total_pipeline_ms": 10.0},
        "bbox_source": "fast",
        "embedding": [0.1] * 512,
    }
    FakeSearchService.top_k_result = [
        {"user_id": 42, "similarity": 0.9},
    ]
    FakeVerificationService.decision = {
        "status": "match", "user_id": 42, "similarity": 0.9,
    }
    _call_task(job_id="job-1", image_url="img-1", user_id="42",
               require_liveness=False)
    done_update = next(u for u in FakeJobRepo.updates if u[1] == JobStatus.done)
    assert done_update[3]["status"] == "match"
    assert done_update[3]["user_id"] == 42
    assert done_update[3]["liveness_passed"] is True
    # лог верификации записан
    assert len(FakeVerificationRepo.logs) == 1
    assert FakeVerificationRepo.logs[0]["success"] is True