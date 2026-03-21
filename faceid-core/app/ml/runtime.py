# faceid-core/app/ml/runtime.py

from functools import lru_cache
from pathlib import Path

import onnxruntime as ort
from insightface.app import FaceAnalysis

from app.core.config import settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(settings.MODELS_DIR)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_MODELS_DIR = PROJECT_ROOT / "models"


def _make_session_options() -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    return so


def _detect_models_root() -> Path:
    """
    Determine the filesystem root that already contains the unzipped
    `buffalo_l` models. If the repo-local `models` folder exists, use it.
    Otherwise, fall back to the configured settings path.
    """
    candidates = [
        PROJECT_MODELS_DIR,
        MODELS_DIR / "models"
    ]

    for candidate in candidates:
        if (candidate / "buffalo_l").exists():
            return candidate.parent if candidate.name == "models" else candidate

    return MODELS_DIR


def get_available_providers():

    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


@lru_cache(maxsize=1)
def get_face_app() -> FaceAnalysis:

    providers = get_available_providers()
    sess_options = _make_session_options()

    ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
    root_dir = _detect_models_root()

    try:
        app = FaceAnalysis(
            name="buffalo_l",
            root=str(root_dir),
            providers=providers,
            sess_options=sess_options,
        )
    except TypeError:
        app = FaceAnalysis(
            name="buffalo_l",
            root=str(root_dir)
        )

    app.prepare(
        ctx_id=ctx_id,
        det_size=(640, 640)
    )

    return app


@lru_cache(maxsize=1)
def get_liveness_model():

    model_path = MODELS_DIR / "antispoof.onnx"

    if not model_path.exists():
        return None

    providers = get_available_providers()
    sess_options = _make_session_options()

    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=providers
    )

    return session
