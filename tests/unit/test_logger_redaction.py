# test_logger_redaction.py — BiometryRedactionFilter + JsonFormatter (152-ФЗ).
#
# Контракт: биометрия (эмбеддинги, кропы лиц, base64-фото, bytes-изображения) не
# должна попадать в JSON-лог. При этом безопасные метаданные/тайминги (image_url,
# align_crop_ms, face_count, similarity, job_id) сохраняются. Маркер 'unit'.
from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from app.core.logger import (
    BiometryRedactionFilter,
    JsonFormatter,
    _is_biometric_key,
    _redact,
    setup_logging,
)

pytestmark = pytest.mark.unit


# --- детектор биометрических ключей ---

@pytest.mark.parametrize(
    "key",
    ["embedding", "EMBEDDING", "query_embedding", "ref_embedding",
     "face_input", "image_bytes", "image_b64", "base64_image",
     "user_embedding", "aligned_crop", "raw_image", "vector", "face_crop"],
)
def test_is_biometric_key_true(key):
    assert _is_biometric_key(key) is True


@pytest.mark.parametrize(
    "key",
    ["image_url", "align_crop_ms", "detect_blob_ms", "face_count",
     "similarity", "job_id", "liveness_score", "bbox", "landmarks",
     "vector_search_ms", "quality_details"],
)
def test_is_biometric_key_false_safe_metadata(key):
    # image_url / align_crop_ms / face_count содержат «image»/«crop»/«face» как
    # подстроку, но это метаданные/тайминги — redact НЕ должен их трогать.
    assert _is_biometric_key(key) is False


# --- _redact: значения ---

def test_redact_ndarray_always_redacted():
    emb = np.zeros(512, dtype=np.float32)
    assert _redact(emb) == "[REDACTED:ndarray(512,)]"


def test_redact_bytes_image():
    assert _redact(b"\xff\xd8\xff\xe0" + b"\x00" * 1000) == "[REDACTED:bytes:1004]"


def test_redact_dict_recurse():
    out = _redact({"embedding": np.zeros(4), "similarity": 0.9, "nested": {"face_input": 1}})
    assert out["embedding"] == "[REDACTED:ndarray(4,)]"
    assert out["similarity"] == 0.9
    assert out["nested"]["face_input"] == "[REDACTED]"


def test_redact_base64_blob_in_string():
    blob = "A" * 400  # ≥256 → base64-блоб
    out = _redact(f"data:image/png;base64,{blob}")
    assert "[REDACTED:b64]" in out
    assert blob not in out


def test_redact_short_base64_preserved():
    # Короткая base64-строка (<256) — не блоб, обычный токен. Не трогаем.
    out = _redact("token=abc123==")
    assert out == "token=abc123=="


# --- BiometryRedactionFilter на LogRecord ---

def _make_record(msg, **extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg=msg, args=None, exc_info=None,
    )
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def test_filter_redacts_biometric_extra_fields():
    record = _make_record(
        "verify done",
        embedding=np.zeros(512, dtype=np.float32),
        face_input=np.zeros((112, 112, 3), dtype=np.uint8),
        image_bytes=b"\x00" * 5000,
        similarity=0.87,
        image_url="verify/job123/file.jpg",
    )
    BiometryRedactionFilter().filter(record)

    assert record.embedding == "[REDACTED:ndarray(512,)]"
    assert record.face_input == "[REDACTED:ndarray(112, 112, 3)]"
    assert record.image_bytes == "[REDACTED:bytes:5000]"
    # безопасные поля не тронуты
    assert record.similarity == 0.87
    assert record.image_url == "verify/job123/file.jpg"


def test_filter_sanitizes_base64_in_message():
    blob = "B" * 400
    record = _make_record(f"image={blob}")
    BiometryRedactionFilter().filter(record)
    assert "[REDACTED:b64]" in record.msg
    assert blob not in record.msg


def test_filter_redacts_args_tuple():
    blob = "C" * 400
    record = _make_record("payload=%s", args=(blob,))
    # LogRecord хранит args; фильтр редактит их до интерполяции.
    BiometryRedactionFilter().filter(record)
    assert record.args == ("[REDACTED:b64]",)


# --- JsonFormatter end-to-end ---

def test_formatter_output_is_json_without_biometry():
    record = _make_record(
        "verify_result",
        embedding=np.zeros(8, dtype=np.float32),
        similarity=0.9,
        job_id="job-1",
        image_url="verify/job-1/file.jpg",
    )
    # Без навешанного filter — formatter сам делает redaction (defense-in-depth).
    line = JsonFormatter().format(record)
    parsed = json.loads(line)

    assert parsed["level"] == "INFO"
    assert parsed["message"] == "verify_result"
    assert parsed["job_id"] == "job-1"
    assert parsed["similarity"] == 0.9
    assert parsed["image_url"] == "verify/job-1/file.jpg"
    assert parsed["embedding"] == "[REDACTED:ndarray(8,)]"
    # сырых чисел эмбеддинга (zeros(8) → "0.0"×8) в выводе нет
    assert "0.0" not in parsed["embedding"]


def test_formatter_preserves_safe_nested_dict():
    record = _make_record(
        "pipeline_completed",
        timings={"align_crop_ms": 3.2, "encode_ms": 15.0, "liveness_ms": 12.1},
        quality_details={"blur_score": 18.4, "face_count": 1},
    )
    line = JsonFormatter().format(record)
    parsed = json.loads(line)
    assert parsed["timings"]["align_crop_ms"] == 3.2
    assert parsed["quality_details"]["face_count"] == 1


# --- setup_logging wiring ---

def test_setup_logging_installs_redaction_filter():
    # Сохраняем предыдущее состояние root, восстанавливаем после.
    root = logging.getLogger()
    prev_handlers = root.handlers
    prev_filters = root.filters
    try:
        setup_logging()
        assert any(isinstance(f, BiometryRedactionFilter) for f in root.filters)
        assert len(root.handlers) == 1
        handler = root.handlers[0]
        assert isinstance(handler.formatter, JsonFormatter)
        assert any(isinstance(f, BiometryRedactionFilter) for f in handler.filters)
    finally:
        root.handlers = prev_handlers
        root.filters = prev_filters