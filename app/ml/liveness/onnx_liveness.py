import logging
from typing import Tuple

import numpy as np
import onnxruntime as ort

from app.core.config import settings
from app.ml.liveness.crop import _softmax, crop_face_square

logger = logging.getLogger("liveness")


class OnnxLivenessChecker:
    """
    Passive liveness detection на yakhyo MiniFASNetV2 (CelebA-Spoof, 3-класс).

    Контракт модели зафиксирован и валидируется при инициализации и в predict:
      - кроп: квадратный crop_face_square(scale=settings.LIVENESS_CROP_SCALE=2.7,
        out_size=80) по bbox детектора — это load-bearing деталь, на ней держится
        измеренный AUC 0.97. НЕ передавайте сюда аффинно-выровненный 112×112 кроп
        и не прямоугольный bbox-crop — точность упадёт.
      - preprocess: 80×80, BGR (без swapRB), 0-255 float32 **БЕЗ /255**, NCHW.
        Размер 80×80 подтверждён по входному тензору модели (['batch_size', 3, 80, 80]).
      - выход: ровно 3 логита формы (1, 3). Конвенция yakhyo MiniFASNet:
        индекс 1 = Real (live), индексы 0/2 = spoof. real_score = softmax(logits)[1].
      - форма выхода, отличная от (1, 3), вызывает RuntimeError (fail-fast),
        чтобы предотвратить молчаливую инверсию классов при смене модели.

    Порог решения НЕ хранится в чекере: predict возвращает сырой real_score,
    а caller применяет settings.LIVENESS_THRESHOLD. Это позволяет менять порог
    без переинициализации сессии и соответствует контракту eval-harness
    (evaluation.liveness.predict.LivenessChecker).
    """

    # Конвенция yakhyo MiniFASNet: idx1 = Real. Захардкожен намеренно —
    # вынос в config создал бы риск молчаливой инверсии при опечатке.
    REAL_IDX = 1

    def __init__(self, model_path: str):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = settings.ONNX_INTRA_OP_THREADS
        sess_options.inter_op_num_threads = settings.ONNX_INTER_OP_THREADS

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

        # Размер входа из тензора модели (ожидается 80), с fallback на config.
        shape = self.session.get_inputs()[0].shape
        if shape and shape[-1] and not isinstance(shape[-1], str):
            self.input_size = int(shape[-1])
        else:
            self.input_size = int(settings.LIVENESS_INPUT_SIZE)
        self.crop_scale = float(settings.LIVENESS_CROP_SCALE)

        # Self-check контракта на этапе инициализации: модель обязана отдавать 3 класса.
        out = self.session.get_outputs()[0]
        out_shape = tuple(out.shape)
        if not out_shape or out_shape[-1] != 3:
            raise RuntimeError(
                f"liveness model must output 3 classes (yakhyo MiniFASNet), got shape {out_shape}"
            )
        logger.info(
            "liveness output: name=%s shape=%s input_size=%d crop_scale=%.2f real_idx=%d",
            out.name, out_shape, self.input_size, self.crop_scale, self.REAL_IDX,
        )

    def _build_tensor(self, image_bgr: np.ndarray, bbox_xyxy) -> np.ndarray | None:
        """
        Кроп + препроцесс в тензор модели (1, 3, input_size, input_size) float32 0-255.
        None если кроп пуст (невалидный bbox).
        """
        crop = crop_face_square(image_bgr, bbox_xyxy, self.crop_scale, self.input_size)
        if crop is None:
            return None
        face = crop.astype(np.float32)            # 0-255, БЕЗ нормализации (контракт yakhyo)
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)
        return np.ascontiguousarray(face, dtype=np.float32)

    def predict_probs(self, image_bgr: np.ndarray, bbox_xyxy) -> Tuple[dict, bool]:
        """То же, что predict, но возвращает per-class вероятности для spoofing_indicators.

        Returns:
            (probs, ok): probs = {"real": float, "spoof": float}. real = softmax[idx1]
            (REAL_IDX), spoof = softmax[idx2]. idx0 — мёртвый класс модели (никогда не
            активируется), в выход не выносим (см. memory liveness-yakhyo-logit-semantics:
            модель бинарная real/spoof, НЕ различает print/replay/cutout). ok=False если
            кроп пуст.
        """
        input_tensor = self._build_tensor(image_bgr, bbox_xyxy)
        if input_tensor is None:
            return {"real": 0.0, "spoof": 0.0}, False

        outputs = self.session.run(None, {self.input_name: input_tensor})
        logits = np.asarray(outputs[0], dtype=np.float32)

        # fail-fast страж против молчаливой инверсии при смене модели.
        if logits.shape != (1, 3):
            raise RuntimeError(
                f"unexpected liveness output shape {logits.shape}, expected (1, 3)"
            )

        probs = _softmax(logits)[0]
        return {"real": float(probs[self.REAL_IDX]), "spoof": float(probs[2])}, True

    def predict(self, image_bgr: np.ndarray, bbox_xyxy) -> Tuple[float, bool]:
        """
        Args:
            image_bgr: полный кадр BGR uint8 (H, W, 3).
            bbox_xyxy: (x1, y1, x2, y2) bbox детектора в координатах кадра.

        Returns:
            (real_score, ok): real_score = softmax(logits)[REAL_IDX] (чем выше,
            тем «живее»); ok=False если кроп пуст (лицо не вырезано).
        """
        probs, ok = self.predict_probs(image_bgr, bbox_xyxy)
        if not ok:
            return 0.0, False
        return probs["real"], True