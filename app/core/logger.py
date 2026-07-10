# app/core/logger.py — структурное JSON-логирование + redaction биометрии (152-ФЗ).
#
# Контракт безопасности: в логи не должны попадать биометрические данные —
# эмбеддинги (512D векторы), исходные фото/кропы лиц, base64-изображения. Логи
# хранятся дольше и шире, чем БД, поэтому redaction здесь — defense-in-depth
# поверх того, что колбэки и так не логируют биометрию явно.
#
# BiometryRedactionFilter применяется к каждому LogRecord до форматирования:
# вычищает биометрические ключи из record.__dict__ (extra-полей) и санитизирует
# record.msg (base64-блобы, длинные float-массивы, ndarray/bytes). JsonFormatter
# дополнительно прогоняет итоговый dict через ту же redaction — на случай, если
# filter обходится (например, при логировании вложенных dict'ов, построенных
# уже после фильтра).

import contextvars
import json
import logging
import re
from datetime import datetime
from typing import Any

try:  # numpy опционален для импорта логгера (не тащим хард-зависимость на момент импорта)
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

# Сквозной trace_id для корреляции логов API → queue → worker.
# Устанавливается request_id_middleware (request_id) и в worker при обработке job.
TRACE_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)


def set_trace_id(trace_id: str | None) -> contextvars.Token[str | None]:
    """Установить trace_id в текущем execution context (async/task/thread)."""
    return TRACE_ID.set(trace_id)


def get_trace_id() -> str | None:
    """Получить текущий trace_id из contextvar."""
    return TRACE_ID.get()


def reset_trace_id(token: contextvars.Token[str | None] | None) -> None:
    """Сбросить trace_id по токену от set_trace_id."""
    if token is not None:
        TRACE_ID.reset(token)


def bind_trace_id(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Дополнить extra-поля trace_id из contextvar (если есть и не задан)."""
    out = dict(extra) if extra else {}
    trace_id = get_trace_id()
    if trace_id is not None and "trace_id" not in out:
        out["trace_id"] = trace_id
    return out


_STANDARD_LOG_RECORD_KEYS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
}

# Точные (case-insensitive) имена биометрических полей. Короткие слова (image,
# crop, face, vector) — только точное совпадение, чтобы не задеть timings-ключи
# вида align_crop_ms / detect_blob_ms / face_count.
_BIOMETRIC_KEYS_EXACT = {
    "image",
    "crop",
    "photo",
    "base64",
    "vector",
    "face",
    "biometric",
    "embedding",
    "embeddings",
    "query_embedding",
    "ref_embedding",
    "embedding_vector",
    "embedding_norm",
    "face_input",
    "face_crop",
    "face_image",
    "aligned_crop",
    "raw_image",
    "image_bytes",
    "image_bgr",
    "image_rgb",
    "image_base64",
    "image_b64",
    "base64_image",
    "input_image",
}

# Подстроки для ловли вариаций (user_embedding, ref_embedding, src_image_bytes…).
# Намеренно узкие: «embed», «face_input», «face_crop», «image_bytes», «image_bgr»,
# «image_rgb», «raw_image», «image_base64», «image_b64», «base64_image»,
# «aligned_crop». Голых «image»/«crop»/«face» здесь НЕТ — иначе заденут image_url,
# align_crop_ms, face_count и т.п. безопасные метаданные/тайминги.
_BIOMETRIC_KEYS_SUBSTR = (
    "embed",
    "face_input",
    "face_crop",
    "face_image",
    "image_bytes",
    "image_bgr",
    "image_rgb",
    "raw_image",
    "image_base64",
    "image_b64",
    "base64_image",
    "aligned_crop",
)

# Длинные base64-блобы (≥256 символов из base64-алфавита) в телах сообщений.
_BASE64_BLOB_RE = re.compile(r"[A-Za-z0-9+/=]{256,}")
# Длинные массивы чисел в теле сообщения (эмбеддинги, отпечатанные через %s).
_NUM_ARRAY_RE = re.compile(r"\[[0-9eE.,\-+\s\n]{200,}\]")


def _is_biometric_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    k = key.lower()
    if k in _BIOMETRIC_KEYS_EXACT:
        return True
    return any(s in k for s in _BIOMETRIC_KEYS_SUBSTR)


def _redact_str(value: str) -> str:
    """Санитизирует строку: вырезает base64-блобы и длинные числовые массивы."""
    if not isinstance(value, str):
        return value
    if _BASE64_BLOB_RE.search(value):
        value = _BASE64_BLOB_RE.sub("[REDACTED:b64]", value)
    if _NUM_ARRAY_RE.search(value):
        value = _NUM_ARRAY_RE.sub("[REDACTED:array]", value)
    return value


def _redact(value: Any, force: bool = False) -> Any:
    """Рекурсивная redaction биометрии в значениях extra-полей и вложенных dict'ах.

    force=True — значение под биометрическим ключом: редактим любой тип (скаляр/
    строку → "[REDACTED]"), но для ndarray/bytes сохраняем type-tag для отладки.
    force=False — обычный прогон: redact только ndarray/bytes/base64-блобы, скаляры
    и безопасные строки проходят как есть.
    """
    # ndarray (эмбеддинг/кроп) — всегда биометрия, независимо от ключа.
    if _np is not None and isinstance(value, _np.ndarray):
        return f"[REDACTED:ndarray{tuple(value.shape)}]"
    if isinstance(value, (bytes, bytearray)):
        return f"[REDACTED:bytes:{len(value)}]"
    if isinstance(value, dict):
        return {
            k: _redact(v, force=_is_biometric_key(k) or force)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(v, force=force) for v in value]
    if isinstance(value, str):
        return "[REDACTED]" if force else _redact_str(value)
    if force:
        return "[REDACTED]"
    return value


class BiometryRedactionFilter(logging.Filter):
    """Вычищает биометрию из LogRecord до форматирования.

    Пробегает record.__dict__ (включая extra-поля), редактит биометрические ключи
    и ndarray/bytes-значения, а также санитизирует record.msg и record.args
    (на случай логирования биометрии прямо в тексте сообщения).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # 1. extra-поля на самом record.
        for key in list(record.__dict__.keys()):
            if key in _STANDARD_LOG_RECORD_KEYS or key.startswith("_"):
                continue
            val = record.__dict__[key]
            record.__dict__[key] = _redact(val, force=_is_biometric_key(key))

        # 2. тело сообщения (могло получить биометрию через %s).
        if isinstance(record.msg, str):
            record.msg = _redact_str(record.msg)
        elif record.msg is not None:
            record.msg = _redact(record.msg)

        # 3. args интерполяции сообщения.
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: ("[REDACTED]" if _is_biometric_key(k) else _redact(v))
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(_redact(a) for a in record.args)

        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_RECORD_KEYS and not key.startswith("_")
        }
        if extra_fields:
            log_record.update(extra_fields)

        # Сквозной trace_id: приоритет у extra-поля, fallback на contextvar.
        if "trace_id" not in log_record:
            trace_id = get_trace_id()
            if trace_id is not None:
                log_record["trace_id"] = trace_id

        # Defense-in-depth: ещё один прогон redaction по итоговому dict —
        # на случай, если BiometryRedactionFilter не навешен на handler/логгер
        # (например, при использовании JsonFormatter в изоляции в тестах).
        log_record = _redact(log_record)

        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log_record, default=str, ensure_ascii=False)


def setup_logging() -> None:
    """Инициализация JSON-логгера с BiometryRedactionFilter на root-логгере."""
    redaction = BiometryRedactionFilter()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.addFilter(redaction)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [handler]
    root_logger.addFilter(redaction)

    # Шумные ML-библиотеки — на WARNING, чтобы не засорять JSON-поток.
    logging.getLogger("insightface").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)