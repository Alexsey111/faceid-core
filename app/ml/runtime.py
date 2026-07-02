# app/ml/runtime.py

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
    det_size: int | None = None,
) -> FaceAnalysis:
    try:
        app = FaceAnalysis(
            name="buffalo_l",
            root=str(root_dir),
            allowed_modules=["detection"],
            providers=providers,
            sess_options=sess_options,
        )
    except TypeError:
        try:
            app = FaceAnalysis(
                name="buffalo_l",
                root=str(root_dir),
                allowed_modules=["detection"],
                providers=providers,
            )
        except TypeError:
            app = FaceAnalysis(
                name="buffalo_l",
                root=str(root_dir),
                allowed_modules=["detection"],
            )
    except Exception:
        app = FaceAnalysis(
            name="buffalo_l",
            root=str(root_dir),
            allowed_modules=["detection"],
        )

    ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
    det_side = max(1, int(det_size if det_size is not None else settings.RETINA_DET_SIZE))
    det_shape = (det_side, det_side)
    app.prepare(ctx_id=ctx_id, det_size=det_shape)
    return app


@lru_cache(maxsize=1)
def get_face_app() -> FaceAnalysis:
    return get_face_app_for_size(int(settings.RETINA_DET_SIZE))


@lru_cache(maxsize=4)
def get_face_app_for_size(det_size: int) -> FaceAnalysis:

    sess_options = _make_session_options()
    root_dir = _detect_models_root()

    for providers in (["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]):
        try:
            return _build_face_app(root_dir, providers, sess_options, det_size=det_size)
        except Exception as exc:
            if providers[0] == "CUDAExecutionProvider":
                logger.warning("CUDA FaceAnalysis init failed, falling back to CPU: %s", exc)
                continue
            raise

    raise RuntimeError("Failed to initialize FaceAnalysis")


@lru_cache(maxsize=1)
def get_liveness_checker():
    """
    Кэшированный OnnxLivenessChecker (production-чекер контракта yakhyo MiniFASNetV2).
    Возвращает None, если модель не найдена или не прошла 3-class guard (тогда
    standalone-пути (/liveness route, celery task) деградируют до 503/ошибки, а
    pipeline_v2 упадёт loud при своём init — это и есть fail-fast для главного пути).
    """
    from app.ml.liveness.onnx_liveness import OnnxLivenessChecker

    model_path = resolve_liveness_model_path(MODELS_DIR)
    if model_path is None:
        return None
    try:
        return OnnxLivenessChecker(str(model_path))
    except Exception as exc:
        logger.warning("liveness checker init failed: %s", exc)
        return None


def _resolve_landmark_106_path() -> Path | None:
    """Путь к 2d106det.onnx (часть pack'а buffalo_l) под MODELS_DIR/PROJECT_MODELS_DIR."""
    rel = Path(getattr(settings, "LIVENESS_LANDMARK_MODEL_REL", "buffalo_l/2d106det.onnx"))
    for cand in (MODELS_DIR, PROJECT_MODELS_DIR):
        p = cand / rel
        if p.is_file():
            return p
    return None


@lru_cache(maxsize=1)
def get_landmarker_106():
    """
    Кэшированный Landmarker106 (2d106det) для active challenge liveness.
    Provider-fallback CUDA → CPU (как у FaceAnalysis). None если модель не найдена
    или все init-попытки упали — caller (challenge-движок) деградирует до 5pt-only.
    """
    from app.ml.liveness.landmarks import Landmarker106

    model_path = _resolve_landmark_106_path()
    if model_path is None:
        logger.warning("landmarker_106 model not found (2d106det.onnx)")
        return None
    so = _make_session_options()
    available = set(ort.get_available_providers())
    for providers in (["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]):
        missing = [p for p in providers if p not in available]
        if missing:
            continue
        try:
            sess = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
            return Landmarker106(str(model_path), session=sess)
        except Exception as exc:
            logger.warning("landmarker_106 init %s failed: %s", providers, exc)
            continue
    try:
        sess = ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])
        return Landmarker106(str(model_path), session=sess)
    except Exception as exc:
        logger.warning("landmarker_106 CPU init failed: %s", exc)
        return None
