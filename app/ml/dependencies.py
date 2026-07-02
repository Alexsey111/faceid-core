from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.ml.batch_encoder import BatchEncoder
from app.ml.embedding.onnx_arcface_encoder import OnnxArcFaceEncoder

_batch_encoder: BatchEncoder | OnnxArcFaceEncoder | None = None


def get_batch_encoder() -> BatchEncoder | OnnxArcFaceEncoder:
    global _batch_encoder
    if _batch_encoder is None:
        model_path = Path(settings.MODELS_DIR) / settings.ARCFACE_MODEL_REL
        encoder = OnnxArcFaceEncoder(model_path)
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
