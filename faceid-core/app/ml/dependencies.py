from __future__ import annotations

from app.core.config import settings
from app.ml.batch_encoder import BatchEncoder
from app.ml.embedding.onnx_arcface_encoder import OnnxArcFaceEncoder

_batch_encoder: BatchEncoder | OnnxArcFaceEncoder | None = None


def get_batch_encoder() -> BatchEncoder | OnnxArcFaceEncoder:
    global _batch_encoder
    if _batch_encoder is None:
        encoder = OnnxArcFaceEncoder()
        if settings.EMBED_BATCH_ENABLED:
            _batch_encoder = BatchEncoder(
                encoder,
                batch_size=settings.EMBED_BATCH_SIZE,
                timeout=settings.EMBED_BATCH_TIMEOUT_MS / 1000.0,
                max_wait_guard_ms=settings.EMBED_BATCH_MAX_WAIT_GUARD_MS,
            )
        else:
            _batch_encoder = encoder
    return _batch_encoder


def reset_batch_encoder() -> None:
    """Reset singleton - useful for testing."""
    global _batch_encoder
    _batch_encoder = None
