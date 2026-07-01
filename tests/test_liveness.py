# test_liveness.py — регрессионные тесты контракта OnnxLivenessChecker
# (yakhyo MiniFASNetV2: 80×80, BGR 0-255 без /255, 3-класс idx1=real, square crop 2.7).
#
# Главный страж — test_live_photo_detected_as_live: на живом фото ожидаем
# real_score > 0.5 (softmax[idx1=real]). При инверсии классов (если взять idx0
# или idx2 как real) живое фото дало бы низкий score, и assert упал бы — это
# защита от возврата инверсии real/fake. Тест НЕ валидирует production-порог
# 0.859 (это дело eval-harness); он ловит именно инверсию + деградацию.

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import settings
from app.ml.liveness.crop import crop_face_square
from app.ml.liveness.model_paths import resolve_liveness_model_path
from app.ml.liveness.onnx_liveness import OnnxLivenessChecker

DATA_DIR = Path(__file__).parent / "data"

LIVE_PHOTO = DATA_DIR / "person1.jpg"
SPOOF_PHOTO = DATA_DIR / "spoof_photo.png"

_MODEL_PATH = resolve_liveness_model_path(Path(settings.MODELS_DIR))
_skip_no_model = pytest.mark.skipif(
    _MODEL_PATH is None,
    reason=f"liveness model not found under {settings.MODELS_DIR}",
)
# Маркер 'unit' отключает миграции БД и flush Redis в conftest — тесты контракта
# чекера не требуют инфраструктуры (см. conftest._all_unit / reset_redis).
_unit = pytest.mark.unit


def _decode(path: Path) -> np.ndarray:
    buf = path.read_bytes()
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        pytest.skip(f"cannot decode image: {path}")
    return img


def _detect_bbox(image: np.ndarray):
    """
    Детектирует top-1 лицо и возвращает его bbox (как в pipeline_v2 / scoring).
    Если детектор недоступен или лиц нет — pytest.skip (нельзя тестировать
    liveness-контракт без bbox; отдавать весь кадр нельзя — это другой вход).
    """
    try:
        from app.ml.detection.retinaface_detector import RetinaFaceDetector

        detector = RetinaFaceDetector(det_size=settings.RETINA_DET_SIZE)
        faces = detector.detect(image) or []
    except Exception:
        faces = []

    if not faces:
        pytest.skip("no face detected — cannot test liveness contract without bbox")
    return faces[0]["bbox"]


@_unit
@_skip_no_model
def test_crop_face_square_contract():
    """crop_face_square отдаёт (out_size, out_size, 3) uint8; центр кропа = центр bbox."""
    image = np.full((200, 200, 3), 100, dtype=np.uint8)
    # белый квадрат 100×100 по центру — bbox (50,50,150,150)
    image[50:150, 50:150] = 255
    crop = crop_face_square(image, (50, 50, 150, 150), scale=2.7, out_size=80)

    assert crop is not None
    assert crop.shape == (80, 80, 3), f"unexpected crop shape {crop.shape}"
    assert crop.dtype == np.uint8
    # центр кропа (40,40) должен быть внутри белого квадрата (центр лица)
    assert crop[40, 40].mean() == 255.0, "crop center should be inside the face region"


@_unit
@_skip_no_model
def test_preprocess_contract():
    """Тензор модели: (1, 3, 80, 80) float32 в [0, 255] (БЕЗ /255) — контракт yakhyo."""
    checker = OnnxLivenessChecker(str(_MODEL_PATH))
    image = np.full((200, 200, 3), 128, dtype=np.uint8)
    tensor = checker._build_tensor(image, (50, 50, 150, 150))

    assert tensor is not None
    assert tensor.shape == (1, 3, 80, 80), f"unexpected shape {tensor.shape}"
    assert tensor.dtype == np.float32
    # контракт yakhyo: 0-255 БЕЗ нормализации
    assert tensor.min() >= 0.0 and tensor.max() <= 255.0


@_unit
@_skip_no_model
def test_output_shape_contract():
    """Выход модели — ровно (1, 3): три логита (yakhyo 3-класс). Дублирует self-check."""
    checker = OnnxLivenessChecker(str(_MODEL_PATH))
    image = np.full((200, 200, 3), 128, dtype=np.uint8)
    tensor = checker._build_tensor(image, (50, 50, 150, 150))
    assert tensor is not None
    outputs = checker.session.run(None, {checker.input_name: tensor})
    logits = np.asarray(outputs[0], dtype=np.float32)
    assert logits.shape == (1, 3), f"unexpected output shape {logits.shape}"


@_unit
@_skip_no_model
def test_live_photo_detected_as_live():
    """
    Регрессионный страж инверсии: живое лицо обязано давать real_score > 0.5
    (softmax[idx1=real]). Порог 0.5 здесь — guard-порог против инверсии классов,
    а НЕ production-порог (0.859 — дело eval-harness/config).
    """
    checker = OnnxLivenessChecker(str(_MODEL_PATH))
    image = _decode(LIVE_PHOTO)
    bbox = _detect_bbox(image)
    real_score, ok = checker.predict(image, bbox)

    assert ok, "liveness crop was empty (predict returned ok=False)"
    assert isinstance(real_score, float)
    assert real_score > 0.5, (
        f"live photo score={real_score:.4f} <= 0.5 — возможна инверсия классов "
        f"(real_idx взят неправильно) или деградация кропа"
    )


@_unit
@_skip_no_model
def test_spoof_photo_runs_with_valid_output():
    """
    SPOOF_PHOTO — нейросетевая генерация (вне распределения CelebA-Spoof),
    поэтому метка не валидируется. Тест лишь прогоняет predict и проверяет
    контракт выхода: (real_score, ok), score в [0, 1].
    """
    checker = OnnxLivenessChecker(str(_MODEL_PATH))
    image = _decode(SPOOF_PHOTO)
    bbox = _detect_bbox(image)
    real_score, ok = checker.predict(image, bbox)

    assert isinstance(ok, bool), f"ok must be bool, got {type(ok)}"
    assert isinstance(real_score, float), f"score must be float, got {type(real_score)}"
    assert 0.0 <= real_score <= 1.0, f"score out of [0,1]: {real_score}"