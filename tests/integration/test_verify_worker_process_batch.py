# tests/integration/test_verify_worker_process_batch.py — integration-покрытие
# основного worker-loop: collect_batch (реальный Redis, brpop/lpop) →
# process_batch (decode MinIO → prepare → encode → search → decision →
# VerifyResultStore в Redis → finalize) для четырёх исходов.
#
# Integration: очередь face_verify_queue и result-store «job:{id}» идут через
# НАСТОЯЩИЙ Redis (localhost:6379, как conftest). Мокаем только ML/DB-слой:
# _PIPELINE, MinioClient, AsyncSessionLocal+repos/VerificationService —
# иначе тест требовал бы GPU-пайплайн и pgvector. Webhook молчит
# (WEBHOOK_ENABLED=False default). finalize_job идёт в реальный Redis.
#
# 152-ФЗ-проверка: во ВСЕХ ветках (ok/quality_reject/spoof/invalid_image)
# исходное фото удаляется из MinIO (finally process_batch → _cleanup_minio_image).

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import cv2
import numpy as np
import pytest

import app.services.verify_job_queue as vjq
import app.services.verify_result_store as vrs
import app.workers.verify_worker as vw
from app.core.config import settings

# Изолированная Redis-БД для теста. Production-worker (compose, db=0) постоянно
# brpop'ает face_verify_queue — на db=0 он мгновенно выгреб бы тестовые job'ы до
# нашего collect_batch (collect_batch завис бы на brpop навсегда). db=15 никто
# не слушает → чистый integration worker-loop без гонки с живым worker'ом.
_TEST_REDIS_DB = 15


def _test_redis_client():
    import redis as _redis
    return _redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=_TEST_REDIS_DB,
        decode_responses=True,
    )


# ------------------------------- helpers --------------------------------

def _valid_jpeg_bytes() -> bytes:
    """Валидный JPEG 100×100 — _decode_image вернёт ndarray (full-res, без downscale)."""
    arr = np.full((100, 100, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", arr)
    assert ok
    return buf.tobytes()


def _job_in_queue(job_id: str, image_url: str = "img-1") -> None:
    """Положить один job в Redis-очередь face_verify_queue (минимальный payload;
    collect_batch дозаполнит dequeued_at/worker_claimed_at_ns/...)."""
    now_ns = int(time.time() * 1_000_000_000)
    job = {
        "job_id": job_id,
        "payload": {
            "image_url": image_url,
            "user_id": "42",
            "require_liveness": False,
            "accepted_at_ns": now_ns,
            "enqueued_at_ns": now_ns,
        },
        "created_at": time.time(),
    }
    vw.redis_client.rpush(vw.QUEUE_NAME, json.dumps(job))


def _read_result_envelope(job_id: str) -> dict[str, Any]:
    """Достать конверт job:{id} из реального Redis (через verify_result_store)."""
    raw = vrs.redis_client.get(f"job:{job_id}")
    assert raw is not None, f"result envelope job:{job_id} not written"
    return json.loads(raw)


# --------------------------- фейковый ML/DB-слой ---------------------------

class FakeMinioClient:
    """get_image → байты (валидный JPEG или битый); delete фиксируется."""
    image_bytes: bytes = b""
    deleted: list[str] = []

    def __init__(self) -> None:
        pass

    def get_image(self, url: str) -> bytes:
        return FakeMinioClient.image_bytes

    def delete_image(self, url: str) -> None:
        FakeMinioClient.deleted.append(url)


class FakePipeline:
    """prepare_face_inputs_from_images → список prepared-dict; encoder.encode_batch
    → список эмбеддингов. prepared задаётся сценарием через класс-атрибут."""
    prepared_results: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []

    def __init__(self) -> None:
        self.fast_detector = type("D", (), {"last_batch_timings": {}})()

        class _Enc:
            last_batch_timings = {"encode_ort_run_ms": 1.0}

            def encode_batch(self_inner, face_inputs):
                return list(FakePipeline.embeddings)

        self.encoder = _Enc()

    def prepare_face_inputs_from_images(self, images):
        return list(FakePipeline.prepared_results)

    def prepare_face_inputs(self, bytes_list):
        return list(FakePipeline.prepared_results)


class FakeAsyncSessionLocal:
    """AsyncSessionLocal() → async ctx-manager c commit()."""

    def __call__(self):
        class _Ctx:
            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *exc):
                return False

            async def commit(self_inner):
                return None

        return _Ctx()


class FakeEmbeddingRepo:
    top_k_batch: list[list[dict]] = []

    def __init__(self, db) -> None:
        self.db = db

    async def find_top_k_batch(self, embeddings, k=2):
        return list(FakeEmbeddingRepo.top_k_batch)


class FakeVerificationRepo:
    def __init__(self, db) -> None:
        self.db = db


class FakeSearchService:
    def __init__(self, embedding_repo) -> None:
        self.embedding_repo = embedding_repo


class FakeVerificationService:
    verify_result: dict[str, Any] = {}

    def __init__(self, embedding_repo=None, verification_repo=None,
                 search_service=None, pipeline=None, load_pipeline=False) -> None:
        pass

    async def verify_from_pipeline_result(self, features, **kwargs):
        return dict(FakeVerificationService.verify_result)


# ------------------------------- фикстура --------------------------------

@pytest.fixture
def worker_env(monkeypatch):
    """Реальный Redis (db=15) для очереди+result-store+finalize; mock только
    ML/DB. Изоляция от production-worker (db=0) — см. _TEST_REDIS_DB."""
    test_redis = _test_redis_client()
    test_redis.flushdb()
    # Все три модуля, в которых redis_client создан на db=0 при импорте,
    # переадресуем на изолированный клиент.
    monkeypatch.setattr(vw, "redis_client", test_redis)
    monkeypatch.setattr(vrs, "redis_client", test_redis)
    monkeypatch.setattr(vjq, "redis_client", test_redis)

    FakeMinioClient.deleted = []
    FakeMinioClient.image_bytes = _valid_jpeg_bytes()
    FakePipeline.prepared_results = []
    FakePipeline.embeddings = [np.ones(512, np.float32)]
    FakeEmbeddingRepo.top_k_batch = []
    FakeVerificationService.verify_result = {}

    monkeypatch.setattr(vw, "MinioClient", FakeMinioClient)
    monkeypatch.setattr(vw, "_PIPELINE", FakePipeline())
    monkeypatch.setattr(vw, "AsyncSessionLocal", FakeAsyncSessionLocal())
    monkeypatch.setattr(vw, "EmbeddingRepository", FakeEmbeddingRepo)
    monkeypatch.setattr(vw, "VerificationRepository", FakeVerificationRepo)
    monkeypatch.setattr(vw, "SearchService", FakeSearchService)
    monkeypatch.setattr(vw, "VerificationService", FakeVerificationService)
    # webhook молчит (default False), но явно для детерминизма
    monkeypatch.setattr(settings, "WEBHOOK_ENABLED", False, raising=False)


def _run_collect_and_process(job_id: str = "j1", image_url: str = "img-1") -> None:
    """Положить job в очередь → collect_batch → process_batch (в event-loop)."""
    _job_in_queue(job_id, image_url=image_url)

    async def _go():
        jobs = await vw.collect_batch()
        assert len(jobs) == 1, f"collect_batch вернул {len(jobs)} job(ов)"
        await vw.process_batch(jobs)

    asyncio.run(_go())


# ------------------------------- сценарии --------------------------------

@pytest.mark.integration
def test_process_batch_ok_match(worker_env):
    """Основной путь: ok → encode → search → verify_from_pipeline_result(match)
    → result-store(done) в Redis. Исходник удалён из MinIO (152-ФЗ)."""
    FakePipeline.prepared_results = [{
        "status": "ok",
        "face_input": object(),
        "bbox": [0, 0, 100, 100],
        "bbox_source": "fast",
        "bbox_source_detail": None,
        "timings": {"detect_ms": 1.0, "liveness_ms": 0.5},
    }]
    FakeEmbeddingRepo.top_k_batch = [[{"user_id": 42, "similarity": 0.9}]]
    FakeVerificationService.verify_result = {
        "status": "match",
        "user_id": 42,
        "similarity": 0.9,
        "liveness_passed": True,
        "timings": {"decision_ms": 1.0, "vector_search_ms": 2.0},
    }

    _run_collect_and_process(job_id="j-ok")

    envelope = _read_result_envelope("j-ok")
    assert envelope["status"] == "done"
    assert envelope["result"]["status"] == "match"
    assert envelope["result"]["user_id"] == 42
    # 152-ФЗ: исходное фото удалено из MinIO после успешной обработки.
    assert "img-1" in FakeMinioClient.deleted


@pytest.mark.integration
def test_process_batch_quality_reject(worker_env):
    """prepared.status='quality_reject' → терминал reject, без encode/search."""
    FakePipeline.prepared_results = [{
        "status": "quality_reject",
        "quality_reason": "face_too_small",
        "quality_details": {"min_face_side": 80},
        "bbox_source": "fast",
        "bbox_source_detail": None,
        "timings": {"detect_ms": 1.0},
    }]

    _run_collect_and_process(job_id="j-qr")

    envelope = _read_result_envelope("j-qr")
    assert envelope["status"] == "done"
    assert envelope["result"]["status"] == "quality_reject"
    assert envelope["result"]["reason"] == "face_too_small"
    assert "img-1" in FakeMinioClient.deleted


@pytest.mark.integration
def test_process_batch_spoof(worker_env):
    """prepared.status='spoof' → spoof_detected (терминал reject), без search."""
    FakePipeline.prepared_results = [{
        "status": "spoof",
        "liveness_score": 0.12,
        "liveness_spoof_score": 0.88,
        "bbox": None,
        "bbox_source": "fast",
        "bbox_source_detail": None,
        "timings": {"liveness_ms": 3.0},
    }]

    _run_collect_and_process(job_id="j-spoof")

    envelope = _read_result_envelope("j-spoof")
    assert envelope["status"] == "done"
    assert envelope["result"]["status"] == "spoof_detected"
    assert envelope["result"]["liveness_passed"] is False
    assert "img-1" in FakeMinioClient.deleted


@pytest.mark.integration
def test_process_batch_invalid_image(worker_env):
    """Битый JPEG (cv2.imdecode → None) → invalid_image терминал error.
    Job не попадает в batch_candidates → encode/search не вызываются."""
    FakeMinioClient.image_bytes = b"not-an-image"

    _run_collect_and_process(job_id="j-bad")

    envelope = _read_result_envelope("j-bad")
    assert envelope["status"] == "done"
    assert envelope["result"]["status"] == "processing_failed"
    assert envelope["result"]["error_code"] == "invalid_image"
    # 152-ФЗ: даже при невалидном payload исходник (если был object_name)
    # удаляется в finally process_batch.
    assert "img-1" in FakeMinioClient.deleted