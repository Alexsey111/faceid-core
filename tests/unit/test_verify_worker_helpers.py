# tests/unit/test_verify_worker_helpers.py — покрытие чистых хелперов
# verify_worker (без Redis/celery-loop). Основной worker-loop (collect_batch/
# process_batch/run_worker) требует живого брокера и покрыт отдельно/интеграционно.

from __future__ import annotations

import numpy as np
import pytest

import app.workers.verify_worker as vw
from app.core.timing import StageTimings


# ------------------------------ _is_job_stale ------------------------------

@pytest.mark.unit
def test_is_job_stale_disabled_by_default():
    """ENABLE_WORKER_EXPIRY=false (default) → никогда не stale, даже при
    большой задержке. Идемпотентность worker'а."""
    assert vw._is_job_stale(0.0, 1000.0) is False


@pytest.mark.unit
def test_is_job_stale_by_queue_wait_sec(monkeypatch):
    monkeypatch.setattr(vw, "ENABLE_WORKER_EXPIRY", True)
    monkeypatch.setattr(vw, "MAX_QUEUE_WAIT_SEC", 10.0)
    monkeypatch.setattr(vw, "MAX_JOB_AGE_MS", 0)
    # задержка 12с > лимита 10с → stale
    assert vw._is_job_stale(0.0, 12.0) is True
    # в пределах лимита → fresh
    assert vw._is_job_stale(0.0, 5.0) is False


@pytest.mark.unit
def test_is_job_stale_by_job_age_ms(monkeypatch):
    """MAX_JOB_AGE_MS > 0 → второй критерий stale (по возрасту в мс)."""
    monkeypatch.setattr(vw, "ENABLE_WORKER_EXPIRY", True)
    monkeypatch.setattr(vw, "MAX_QUEUE_WAIT_SEC", 0)
    monkeypatch.setattr(vw, "MAX_JOB_AGE_MS", 5000)  # 5с
    assert vw._is_job_stale(0.0, 6.0) is True   # 6000ms > 5000
    assert vw._is_job_stale(0.0, 4.0) is False


# ------------------------- _build_technical_timestamps -------------------------

@pytest.mark.unit
def test_build_technical_timestamps_filters_none_and_converts():
    """None-аргументы пропускаются; единицы конвертируются (ns→ms, s→ms)."""
    out = vw._build_technical_timestamps(
        queue_popped_at=1.5,            # секунды → 1500 ms
        worker_started_at_ns=2_000_000,  # ns → 2 ms
        completed_at_ns=3_000_000,        # ns → 3 ms
    )
    assert out == {
        "queue_popped_at_ms": 1500,
        "worker_started_at_ms": 2,
        "completed_at_ms": 3,
    }
    # без аргументов → пустой dict
    assert vw._build_technical_timestamps() == {}


# --------------------------- _payload_timestamp_ns ---------------------------

@pytest.mark.unit
def test_payload_timestamp_ns_fallback_on_none():
    assert vw._payload_timestamp_ns({}, "missing", 99) == 99


@pytest.mark.unit
def test_payload_timestamp_ns_valid_int():
    assert vw._payload_timestamp_ns({"accepted_at_ns": 42}, "accepted_at_ns", 0) == 42


@pytest.mark.unit
def test_payload_timestamp_ns_invalid_returns_fallback():
    assert vw._payload_timestamp_ns({"accepted_at_ns": "not-a-number"}, "accepted_at_ns", 7) == 7


# ----------------------- _normalize_job_stage_timings -----------------------

@pytest.mark.unit
def test_normalize_job_stage_timings_defaults_and_alias():
    """Пустой/None → все нули; fast_detect_ms/align_crop_ms/vector_search_ms —
    алиасы для detect_ms/align_ms/search_ms; pipeline_total_ms = сумма стадий."""
    out = vw._normalize_job_stage_timings(None)
    assert out["preprocess_ms"] == 0.0
    assert out["pipeline_total_ms"] == 0.0

    out = vw._normalize_job_stage_timings({
        "fast_detect_ms": 5.0,
        "align_crop_ms": 3.0,
        "vector_search_ms": 8.0,
        "encode_ms": 4.0,
    })
    assert out["detect_ms"] == 5.0      # алиас
    assert out["align_ms"] == 3.0       # алиас
    assert out["search_ms"] == 8.0      # алиас
    # pipeline_total = preprocess+detect+align+encode+search+liveness+decision
    assert out["pipeline_total_ms"] == 5.0 + 3.0 + 8.0 + 4.0


# --------------------------- _build_result_payload ---------------------------

@pytest.mark.unit
def test_build_result_payload_merges_timings():
    """result.timings + StageTimings.values сливаются; timestamps добавляются."""
    timings = StageTimings()
    timings.set("queue_wait_ms", 12.0)
    timings.set("search_ms", 8.0)
    result = {"status": "match", "timings": {"preprocess_ms": 1.0}}
    payload = vw._build_result_payload(result, timings, {"accepted_at_ns": 10})
    assert payload["status"] == "match"
    assert payload["timings"]["preprocess_ms"] == 1.0   # из result
    assert payload["timings"]["queue_wait_ms"] == 12.0   # из StageTimings
    assert payload["timings"]["search_ms"] == 8.0
    assert payload["timestamps"] == {"accepted_at_ns": 10}


# ------------------------------ _build_metrics ------------------------------

@pytest.mark.unit
def test_build_metrics_defaults_dequeued_to_started():
    """dequeued_at=None → effective = started_at; задержки в секундах, clamped ≥0."""
    m = vw._build_metrics(created_at=10.0, started_at=12.0, finished_at=15.0)
    assert m["dequeued_at"] == 12.0
    assert m["queue_delay"] == 2.0       # 12 - 10
    assert m["dequeue_to_start"] == 0.0   # 12 - 12
    assert m["processing_time"] == 3.0   # 15 - 12
    assert m["total_latency"] == 5.0      # 15 - 10


@pytest.mark.unit
def test_build_metrics_with_explicit_dequeued():
    m = vw._build_metrics(10.0, 13.0, 16.0, dequeued_at=11.0)
    assert m["dequeued_at"] == 11.0
    assert m["queue_delay"] == 1.0        # 11 - 10
    assert m["dequeue_to_start"] == 2.0   # 13 - 11


# ------------------------------ _decode_image ------------------------------
# Воркер больше не даунскейлит: pipeline хранит original (кроп лица/occ/embedding/
# liveness из full-res) и сам даунскейлит только кадр для детекции.

@pytest.mark.unit
def test_decode_image_invalid_bytes_returns_none():
    """Битые байты → cv2.imdecode вернёт None → (None, timings)."""
    image, timings = vw._decode_image(b"not-an-image")
    assert image is None
    assert "image_decode_ms" in timings
    # ключи сохранены как 0.0 — их читают метрики/логи worker-pre stage
    assert timings["downscale_ms"] == 0.0
    assert timings["jpeg_reencode_ms"] == 0.0


@pytest.mark.unit
def test_decode_image_keeps_full_resolution():
    """Валидное JPEG декодируется в full-res, БЕЗ ресайза (крупный кадр не режется)."""
    import cv2
    arr = np.full((600, 800, 3), 128, dtype=np.uint8)  # 800px long side (>480)
    ok, buf = cv2.imencode(".jpg", arr)
    assert ok
    image, timings = vw._decode_image(buf.tobytes())
    assert image is not None
    assert image.shape == (600, 800, 3)  # full-res сохранён, downscale не делается
    assert timings["downscale_ms"] == 0.0
    assert "image_decode_ms" in timings


# -------------------------- _cleanup_minio_image --------------------------

@pytest.mark.unit
def test_cleanup_minio_image_noop_on_none():
    """Нет image_url (legacy/битый payload) → молча возвращается."""
    # не должен обращаться к MinioClient
    vw._cleanup_minio_image(None, "job-1")
    vw._cleanup_minio_image("", "job-1")


@pytest.mark.unit
def test_cleanup_minio_image_swallows_error(monkeypatch):
    """Ошибка удаления MinIO не пробрасывается (best-effort; lifecycle-cover)."""
    class _BoomMinio:
        def __init__(self):
            pass

        def delete_image(self, url):
            raise RuntimeError("minio gone")

    monkeypatch.setattr(vw, "MinioClient", _BoomMinio)
    # не должен падать
    vw._cleanup_minio_image("img-1", "job-1")


# ------------------------ _prepare_face_inputs_sync ------------------------

@pytest.mark.unit
def test_prepare_face_inputs_sync_decoded_path(monkeypatch):
    """Все items с image (np.ndarray) → prepare_face_inputs_from_images."""
    calls = {}

    class _Pipe:
        fast_detector = None

        def prepare_face_inputs_from_images(self, images):
            calls["from_images"] = list(images)
            return [{"prepared": True}]

        def prepare_face_inputs(self, bytes_list):
            calls["from_bytes"] = list(bytes_list)
            return []

    prepared, prep_ms, det_timings = vw._prepare_face_inputs_sync(
        _Pipe(), [{"image": np.zeros((4, 4, 3), np.uint8)}]
    )
    assert prepared == [{"prepared": True}]
    assert "from_images" in calls and "from_bytes" not in calls
    assert prep_ms >= 0.0


@pytest.mark.unit
def test_prepare_face_inputs_sync_bytes_path(monkeypatch):
    """Если image is None (битый decode) → fallback на prepare_face_inputs
    по image_bytes (152-ФЗ — байты только в памяти)."""
    calls = {}

    class _Pipe:
        fast_detector = None

        def prepare_face_inputs_from_images(self, images):
            calls["from_images"] = images
            return []

        def prepare_face_inputs(self, bytes_list):
            calls["from_bytes"] = list(bytes_list)
            return [{"prepared": True}]

    prepared, _, _ = vw._prepare_face_inputs_sync(
        _Pipe(), [{"image": None, "image_bytes": b"jpg1"}]
    )
    assert prepared == [{"prepared": True}]
    assert "from_bytes" in calls and "from_images" not in calls


# --------------------------- _encode_batch_sync ---------------------------

@pytest.mark.unit
def test_encode_batch_sync_invokes_encoder():
    class _Enc:
        last_batch_timings = {"ort_run_ms": 5.0}

        def encode_batch(self, face_inputs):
            return [np.ones(512, np.float32) for _ in face_inputs]

    class _Pipe:
        encoder = _Enc()

    embeddings, encode_ms, batch_timings = vw._encode_batch_sync(_Pipe(), [object(), object()])
    assert len(embeddings) == 2
    assert batch_timings["ort_run_ms"] == 5.0
    assert encode_ms >= 0.0


# --------------------------- _reject_stale_job ---------------------------

@pytest.mark.unit
def test_reject_stale_job_marks_expired(monkeypatch):
    """Stale job → set_expired + _complete_terminal_job_inline(expired).
    Мокаем side-effect (Redis/result-store), проверяем что путь не падает."""
    calls = {}

    monkeypatch.setattr(vw.VerifyResultStore, "set_expired", lambda *a, **kw: None)
    monkeypatch.setattr(vw, "_complete_terminal_job_inline",
                        lambda **kw: calls.update(kw))
    # метрика — заглушка
    class _M:
        def labels(self, **kw):
            return self

        def inc(self, amount=1):
            return None

    monkeypatch.setattr(vw, "VERIFY_REJECTED_JOBS", _M(), raising=False)

    vw._reject_stale_job("job-stale", created_at=0.0, observed_at=1.0)
    assert calls.get("terminal_state") == "expired"
    assert calls.get("job_id") == "job-stale"
    assert calls.get("outcome") == "expired"