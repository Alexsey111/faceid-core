# faceid-core/app/ml/runtime.py

import os
import logging
from functools import lru_cache
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import onnxruntime as ort

from app.ml.liveness.model_paths import resolve_liveness_model_path
from insightface.app import FaceAnalysis

from app.core.config import settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(settings.MODELS_DIR)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_MODELS_DIR = PROJECT_ROOT / "models"
logger = logging.getLogger(__name__)


def _make_session_options() -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, int(settings.ONNX_INTRA_OP_THREADS))
    so.inter_op_num_threads = max(1, int(settings.ONNX_INTER_OP_THREADS))
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
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
    return ort.get_available_providers()


def _build_face_app(
    root_dir: Path,
    providers: list[str],
    sess_options: ort.SessionOptions,
) -> FaceAnalysis:
    try:
        app = FaceAnalysis(
            name="buffalo_l",
            root=str(root_dir),
            providers=providers,
            sess_options=sess_options,
            det_name="scrfd_10g",
        )
    except TypeError:
        try:
            app = FaceAnalysis(
                name="buffalo_l",
                root=str(root_dir),
                providers=providers,
            )
        except TypeError:
            app = FaceAnalysis(
                name="buffalo_l",
                root=str(root_dir),
            )
    except Exception:
        app = FaceAnalysis(
            name="buffalo_l",
            root=str(root_dir),
        )

    ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
    det_side = max(1, int(settings.RETINA_DET_SIZE))
    det_size = (det_side, det_side)
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app


@lru_cache(maxsize=1)
def get_face_app() -> FaceAnalysis:

    sess_options = _make_session_options()
    root_dir = _detect_models_root()

    for providers in (["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]):
        try:
            return _build_face_app(root_dir, providers, sess_options)
        except Exception as exc:
            if providers[0] == "CUDAExecutionProvider":
                logger.warning("CUDA FaceAnalysis init failed, falling back to CPU: %s", exc)
                continue
            raise

    raise RuntimeError("Failed to initialize FaceAnalysis")


@lru_cache(maxsize=1)
def get_liveness_model():

    model_path = resolve_liveness_model_path(MODELS_DIR)

    if model_path is None:
        return None

    sess_options = _make_session_options()

    for providers in (["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]):
        try:
            return ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=providers
            )
        except Exception as exc:
            if providers[0] == "CUDAExecutionProvider":
                logger.warning("CUDA liveness session init failed, falling back to CPU: %s", exc)
                continue
            raise
