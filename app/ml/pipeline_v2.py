# app\ml\pipeline_v2.py

from typing import Dict, Any, Optional
import logging
import os
import time

import cv2
import numpy as np

from app.core.timing import StageTimings, now_perf_ns
from app.core.config import settings
from app.ml.batch_encoder import BatchEncoder
from app.ml.dependencies import get_batch_encoder
from app.ml.embedding.onnx_arcface_encoder import OnnxArcFaceEncoder
from app.ml.liveness.onnx_liveness import OnnxLivenessChecker
from app.ml.liveness.model_paths import resolve_liveness_model_path
from app.ml.preprocessing.image_preprocessor import ImagePreprocessor
from app.ml.quality.image_quality_gate import ImageQualityGate
from app.ml.detection.retinaface_detector import RetinaFaceDetector
from insightface.utils.face_align import norm_crop
from app.monitoring.metrics import observe_pipeline_stage


logger = logging.getLogger(__name__)

PIPELINE_STAGE_NAMES = (
    "preprocess_ms",
    "detect_ms",
    "align_ms",
    "encode_ms",
    "search_ms",
    "liveness_ms",
    "decision_ms",
)


def expand_bbox(bbox, scale, image_shape):
    x1, y1, x2, y2 = bbox
    h, w = image_shape[:2]

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale

    nx1 = max(0, int(cx - bw / 2))
    ny1 = max(0, int(cy - bh / 2))
    nx2 = min(w, int(cx + bw / 2))
    ny2 = min(h, int(cy + bh / 2))

    return nx1, ny1, nx2, ny2


def crop_image(image, bbox):
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2], (x1, y1)


def scale_faces_to_original(
    faces: list[dict],
    orig_shape: tuple[int, ...],
    ds_shape: tuple[int, ...],
) -> list[dict]:
    """Пересчитать bbox/landmarks лиц из координат downscaled-кадра в координаты original.

    Детекция идёт на downscaled (long-side ≤ max_side, быстро); кроп лица/occ/embedding/
    liveness берутся из original (full-res, без потери разрешения на 16-МП фото).
    sx/sy — независимый масштаб по осям (сохраняет пропорции, т.к. resize равномерный,
    но считаем отдельно для устойчивости к округлению). Возвращает новый список
    словарей, не мутирует вход.
    """
    orig_h, orig_w = orig_shape[:2]
    ds_h, ds_w = ds_shape[:2]
    sx = orig_w / float(ds_w)
    sy = orig_h / float(ds_h)
    scaled: list[dict] = []
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        new_face = dict(face)
        new_face["bbox"] = [
            float(x1) * sx,
            float(y1) * sy,
            float(x2) * sx,
            float(y2) * sy,
        ]
        landmarks = face.get("landmarks")
        if landmarks is not None:
            lm = np.asarray(landmarks, dtype=np.float32).copy()
            if lm.ndim == 2 and lm.shape[1] >= 2:
                lm[:, 0] *= sx
                lm[:, 1] *= sy
            else:
                # Плоский [x0,y0,x1,y1,...] — чётные индексы X, нечётные Y.
                lm[0::2] *= sx
                lm[1::2] *= sy
            new_face["landmarks"] = lm
        scaled.append(new_face)
    return scaled


def _finalize_timings(timings: Dict[str, float]) -> Dict[str, float]:
    finalized = dict(timings)
    finalized["preprocess_ms"] = float(finalized.get("preprocess_ms", 0.0))
    finalized["detect_ms"] = float(finalized.get("detect_ms", finalized.get("fast_detect_ms", 0.0)))
    finalized["align_ms"] = float(finalized.get("align_ms", finalized.get("align_crop_ms", 0.0)))
    finalized["encode_ms"] = float(finalized.get("encode_ms", 0.0))
    finalized["search_ms"] = float(finalized.get("search_ms", 0.0))
    finalized["liveness_ms"] = float(finalized.get("liveness_ms", 0.0))
    if "decision_ms" not in finalized:
        finalized["decision_ms"] = float(
            finalized.get("quality_gate_pre_ms", 0.0) + finalized.get("quality_gate_face_ms", 0.0)
        )
    finalized["total_pipeline_ms"] = sum(float(finalized.get(stage, 0.0)) for stage in PIPELINE_STAGE_NAMES)
    return finalized


def _observe_timings(timings: Dict[str, float]) -> None:
    for stage in PIPELINE_STAGE_NAMES:
        observe_pipeline_stage(stage.replace("_ms", ""), float(timings.get(stage, 0.0) or 0.0))


class FacePipelineV2:
    """
    Production-oriented V2 pipeline:

    1. preprocess
    2. fast detector
    3. crop
    4. ONNX ArcFace encode
    5. search / decision
    """

    FAST_CONFIDENCE_THRESHOLD = 0.75
    FAST_BBOX_EXPAND_SCALE = 0.30

    def __init__(self):
        print(f"PID={os.getpid()} pipeline_v2 created")
        self._initialized = False

        self.preprocessor = ImagePreprocessor()
        self.quality_gate = ImageQualityGate()

        self.fast_detector: Optional[RetinaFaceDetector] = None
        self.encoder: BatchEncoder | OnnxArcFaceEncoder | None = None
        self.liveness_checker: Optional[OnnxLivenessChecker] = None

    def _init(self):
        if not self._initialized:
            # SCRFD (insightface buffalo_l) — отдаёт 5-point landmarks,
            # необходимые для аффинного выравнивания лица перед encode.
            # Грузится внутри runtime.get_face_app_for_size; отдельный
            # поиск путей к caffemodel не требуется.
            self.fast_detector = RetinaFaceDetector(det_size=settings.RETINA_DET_SIZE)
            self.encoder = get_batch_encoder()
            if settings.LIVENESS_ENABLED:
                liveness_path = resolve_liveness_model_path(settings.MODELS_DIR)

                if liveness_path is not None:
                    logger.info("liveness model loaded from %s", liveness_path)
                    # Порог решения применяет caller (ниже в _finalize), не чекер.
                    self.liveness_checker = OnnxLivenessChecker(str(liveness_path))
                else:
                    logger.warning("liveness model not found, disabled")
                    self.liveness_checker = None
            self._initialized = True

    def _prepare_face_from_detection(
        self,
        image: np.ndarray,
        image_quality: Any,
        faces: list[dict],
        timings: Dict[str, float],
    ) -> Dict[str, Any]:
        detection: Dict[str, Any] | None = None
        face_input: np.ndarray | None = None
        bbox_source: str | None = "fast"
        bbox_source_detail: str | None = "fast"

        if len(faces) == 1:
            face = faces[0]
            bbox = face["bbox"]
            confidence = face["confidence"]
            landmarks = face.get("landmarks")
            if confidence < self.FAST_CONFIDENCE_THRESHOLD:
                raise ValueError("Low confidence face detection")
            # raw bbox детектора (до expand) — для liveness. yakhyo MiniFASNet
            # обучен на квадратном кропе crop_face_square(scale=2.7) по raw bbox;
            # checker сам делает кроп, сюда передаём только координаты.
            raw_bbox = tuple(float(v) for v in bbox)
            x1, y1, x2, y2 = map(int, bbox)

            # bbox-crop (расширенный) — для проверки mean.
            # При наличии landmarks encode использует аффинно-выровненный кроп,
            # а не этот roi (чёрные поля warpAffine не должны попадать в liveness).
            # liveness больше не использует этот roi — checker делает свой square-кроп.
            x1, y1, x2, y2 = expand_bbox(
                (x1, y1, x2, y2),
                self.FAST_BBOX_EXPAND_SCALE,
                image.shape,
            )

            t_align_crop = now_perf_ns()
            roi, _ = crop_image(image, (x1, y1, x2, y2))

            if roi.mean() < 20:
                raise ValueError("bad crop")

            bbox_source = "fast"
            bbox_source_detail = "fast"
            # Аффинное выравнивание по 5-point landmarks к шаблону ArcFace 112×112.
            # Fallback на bbox-resize, если landmarks вдруг отсутствуют.
            if landmarks is not None:
                face_input = norm_crop(
                    image,
                    np.asarray(landmarks, dtype=np.float32),
                    112,
                )
            else:
                face_input = cv2.resize(roi, (112, 112))
            timings["align_crop_ms"] = (now_perf_ns() - t_align_crop) / 1_000_000
            timings["align_ms"] = timings["align_crop_ms"]
            detection = {
                "bbox": [x1, y1, x2, y2],
                "landmarks": landmarks,
                "confidence": float(confidence),
            }

        elif len(faces) == 0:
            raise ValueError("Face not detected")

        else:
            raise ValueError("Multiple faces not allowed")

        if detection is None or face_input is None:
            raise ValueError("Failed to prepare face input")

        # ----------------------------
        # POST-DETECT QUALITY GATE
        # ----------------------------
        t0 = now_perf_ns()
        face_quality = self.quality_gate.evaluate_detection(
            bbox=detection["bbox"],
            landmarks=detection.get("landmarks"),
            image=image,
        )
        timings["quality_gate_face_ms"] = (now_perf_ns() - t0) / 1_000_000

        if not face_quality.passed:
            finalized_timings = _finalize_timings(timings)
            # Окклюзия (маска/очки) → status="retry" (просим снять и пере-снять),
            # capture-качество → status="quality_reject". Решение по reason-коду.
            quality_status = (
                "retry" if face_quality.reason == "remove_occlusion" else "quality_reject"
            )
            return {
                "status": quality_status,
                "quality_reason": face_quality.reason,
                "quality_warning": face_quality.details.get("quality_warning"),
                "quality_mode": face_quality.details.get("quality_gate_mode"),
                "quality_details": {
                    **image_quality.details,
                    **face_quality.details,
                },
                "bbox": detection["bbox"],
                "landmarks": detection.get("landmarks"),
                "bbox_source": bbox_source,
                "bbox_source_detail": bbox_source_detail,
                "timings": finalized_timings,
            }

        if face_input.size == 0:
            raise ValueError("Empty face crop")

        if face_input.shape[:2] != (112, 112):
            face_input = cv2.resize(face_input, (112, 112))

        face_input = np.ascontiguousarray(face_input, dtype=np.uint8)

        liveness_passed = None
        liveness_score = None
        liveness_spoof_score = None
        if settings.LIVENESS_ENABLED and self.liveness_checker:
            try:
                t0 = now_perf_ns()
                # liveness: checker сам делает квадратный кроп crop_face_square
                # по raw bbox детектора (контракт yakhyo MiniFASNet, на нём AUC 0.97).
                # НЕ передаём аффинно-выровненный face_input или прямоугольный roi —
                # это сломает точность.
                probs, ok = self.liveness_checker.predict_probs(image, raw_bbox)
                timings["liveness_ms"] = (now_perf_ns() - t0) / 1_000_000
                liveness_score = probs["real"]
                liveness_spoof_score = probs["spoof"]
                liveness_passed = ok and liveness_score >= settings.LIVENESS_THRESHOLD

                if liveness_passed is False:
                    finalized_timings = _finalize_timings(timings)

                    return {
                        "status": "spoof",
                        "liveness_passed": False,
                        "liveness_score": liveness_score,
                        "liveness_spoof_score": liveness_spoof_score,
                        "bbox": detection["bbox"],
                        "landmarks": detection.get("landmarks"),
                        "bbox_source": bbox_source,
                        "bbox_source_detail": bbox_source_detail,
                        "timings": finalized_timings,
                    }

            except Exception as e:
                logger.warning("liveness_error: %s", str(e))
                liveness_passed = None
                liveness_score = None
                liveness_spoof_score = None

        result = {
            "status": "ok",
            "face_input": face_input,
            "bbox": detection["bbox"],
            "landmarks": detection.get("landmarks"),
            "liveness_passed": liveness_passed,
            "liveness_score": liveness_score,
            "liveness_spoof_score": liveness_spoof_score,
            "bbox_source_detail": bbox_source_detail,
            "quality_warning": (
                image_quality.details.get("quality_warning")
                or face_quality.details.get("quality_warning")
            ),
            "quality_mode": (
                face_quality.details.get("quality_gate_mode")
                or image_quality.details.get("quality_gate_mode")
            ),
            "quality_details": {
                **image_quality.details,
                **face_quality.details,
            },
            "timings": _finalize_timings(timings),
        }
        result["bbox_source"] = bbox_source
        return result

    def prepare_face_input(self, image_bytes: bytes) -> Dict[str, Any]:
        self._init()

        assert self.fast_detector is not None, "detector not initialized"
        assert self.encoder is not None, "encoder not initialized"

        stage_timings = StageTimings()
        timings: Dict[str, float] = stage_timings.values

        t0 = now_perf_ns()
        # original — full-res (кроп лица/occ/embedding/liveness из него, чтобы не
        # терять разрешение на 16-МП фото); downscaled ≤ max_side — для быстрой детекции.
        original, downscaled = self.preprocessor.decode_pair(image_bytes)
        stage_timings.finish("preprocess_ms", t0)

        t0 = now_perf_ns()
        # Pre-gate (blur/brightness) на downscaled — глобальные метрики не зависят
        # от разрешения; bbox здесь ещё не нужен.
        image_quality = self.quality_gate.evaluate_image(downscaled)
        timings["quality_gate_pre_ms"] = (now_perf_ns() - t0) / 1_000_000

        if not image_quality.passed:
            finalized_timings = _finalize_timings(timings)
            return {
                "status": "quality_reject",
                "quality_reason": image_quality.reason,
                "quality_warning": image_quality.details.get("quality_warning"),
                "quality_mode": image_quality.details.get("quality_gate_mode"),
                "quality_details": image_quality.details,
                "timings": finalized_timings,
            }

        t0 = now_perf_ns()
        # Детекция на downscaled (быстро) → пересчёт bbox/landmarks в координаты original.
        fast_faces_ds = self.fast_detector.detect(downscaled) or []
        detect_ms = stage_timings.finish("detect_ms", t0)
        timings["fast_detect_ms"] = detect_ms

        fast_faces = scale_faces_to_original(
            fast_faces_ds, original.shape, downscaled.shape
        )
        return self._prepare_face_from_detection(
            original, image_quality, fast_faces, timings
        )

    def prepare_face_inputs(self, image_bytes_list: list[bytes]) -> list[Dict[str, Any]]:
        self._init()

        assert self.fast_detector is not None, "detector not initialized"

        originals: list[np.ndarray] = []
        downscaleds: list[np.ndarray] = []
        image_qualities: list[Any] = []
        timings_list: list[Dict[str, float]] = []
        results: list[Dict[str, Any]] = []
        eligible_indices: list[int] = []
        eligible_downscaled: list[np.ndarray] = []

        for image_bytes in image_bytes_list:
            timings: Dict[str, float] = {}
            t0 = time.time()
            original, downscaled = self.preprocessor.decode_pair(image_bytes)
            timings["preprocess_ms"] = (time.time() - t0) * 1000

            t0 = time.time()
            # Pre-gate на downscaled (глобальные метрики).
            image_quality = self.quality_gate.evaluate_image(downscaled)
            timings["quality_gate_pre_ms"] = (time.time() - t0) * 1000

            originals.append(original)
            downscaleds.append(downscaled)
            image_qualities.append(image_quality)
            timings_list.append(timings)
            if image_quality.passed:
                eligible_indices.append(len(originals) - 1)
                eligible_downscaled.append(downscaled)

        if eligible_downscaled:
            t0 = time.perf_counter()
            eligible_fast_faces_list = self.fast_detector.detect_batch(eligible_downscaled)
            detect_batch_ms = (time.perf_counter() - t0) * 1000.0
            per_image_detect_ms = detect_batch_ms / max(1, len(eligible_downscaled))
        else:
            eligible_fast_faces_list = []
            detect_batch_ms = 0.0
            per_image_detect_ms = 0.0

        if len(eligible_fast_faces_list) != len(eligible_downscaled):
            raise RuntimeError(
                "detect_batch returned unexpected batch size: "
                f"{len(eligible_fast_faces_list)} != {len(eligible_downscaled)}"
            )

        logger.info(
            "[BATCH DETECT] size=%s detect_batch_ms=%.2f per_image_detect_ms=%.2f",
            len(eligible_downscaled),
            detect_batch_ms,
            per_image_detect_ms,
        )

        eligible_fast_faces_by_index = {
            idx: fast_faces
            for idx, fast_faces in zip(eligible_indices, eligible_fast_faces_list)
        }

        for idx, (original, downscaled, image_quality, timings) in enumerate(
            zip(originals, downscaleds, image_qualities, timings_list)
        ):
            if not image_quality.passed:
                finalized_timings = _finalize_timings(timings)
                _observe_timings(finalized_timings)
                results.append(
                    {
                        "status": "quality_reject",
                        "quality_reason": image_quality.reason,
                        "quality_warning": image_quality.details.get("quality_warning"),
                        "quality_mode": image_quality.details.get("quality_gate_mode"),
                        "quality_details": image_quality.details,
                        "timings": finalized_timings,
                    }
                )
                continue

            timings["detect_ms"] = per_image_detect_ms
            timings["detect_batch_ms_total"] = detect_batch_ms
            fast_faces_ds = eligible_fast_faces_by_index.get(idx, [])
            fast_faces = scale_faces_to_original(
                fast_faces_ds, original.shape, downscaled.shape
            )
            result = self._prepare_face_from_detection(
                original, image_quality, fast_faces, timings
            )
            _observe_timings(result["timings"])
            results.append(result)

        return results

    def prepare_face_inputs_from_images(self, images: list[np.ndarray]) -> list[Dict[str, Any]]:
        self._init()

        assert self.fast_detector is not None, "detector not initialized"

        image_qualities: list[Any] = []
        timings_list: list[Dict[str, float]] = []
        downscaleds: list[np.ndarray] = []
        results: list[Dict[str, Any]] = []
        eligible_indices: list[int] = []
        eligible_downscaled: list[np.ndarray] = []

        # images — уже original ndarray (без downscale; worker отдаёт full-res,
        # чтобы кроп лица/occ/embedding/liveness не теряли разрешение). downscale
        # здесь нужен только как кадр для быстрой детекции.
        for original in images:
            timings: Dict[str, float] = {}
            t0 = time.time()
            downscaled = self.preprocessor.process_image(original)
            timings["preprocess_ms"] = (time.time() - t0) * 1000

            t0 = time.time()
            image_quality = self.quality_gate.evaluate_image(downscaled)
            timings["quality_gate_pre_ms"] = (time.time() - t0) * 1000

            downscaleds.append(downscaled)
            image_qualities.append(image_quality)
            timings_list.append(timings)
            if image_quality.passed:
                eligible_indices.append(len(downscaleds) - 1)
                eligible_downscaled.append(downscaled)

        if eligible_downscaled:
            t0 = time.perf_counter()
            eligible_fast_faces_list = self.fast_detector.detect_batch(eligible_downscaled)
            detect_batch_ms = (time.perf_counter() - t0) * 1000.0
            per_image_detect_ms = detect_batch_ms / max(1, len(eligible_downscaled))
        else:
            eligible_fast_faces_list = []
            detect_batch_ms = 0.0
            per_image_detect_ms = 0.0

        if len(eligible_fast_faces_list) != len(eligible_downscaled):
            raise RuntimeError(
                "detect_batch returned unexpected batch size: "
                f"{len(eligible_fast_faces_list)} != {len(eligible_downscaled)}"
            )

        logger.info(
            "[BATCH DETECT] size=%s detect_batch_ms=%.2f per_image_detect_ms=%.2f",
            len(eligible_downscaled),
            detect_batch_ms,
            per_image_detect_ms,
        )

        eligible_fast_faces_by_index = {
            idx: fast_faces
            for idx, fast_faces in zip(eligible_indices, eligible_fast_faces_list)
        }

        for idx, (original, downscaled, image_quality, timings) in enumerate(
            zip(images, downscaleds, image_qualities, timings_list)
        ):
            if not image_quality.passed:
                finalized_timings = _finalize_timings(timings)
                _observe_timings(finalized_timings)
                results.append(
                    {
                        "status": "quality_reject",
                        "quality_reason": image_quality.reason,
                        "quality_warning": image_quality.details.get("quality_warning"),
                        "quality_mode": image_quality.details.get("quality_gate_mode"),
                        "quality_details": image_quality.details,
                        "timings": finalized_timings,
                    }
                )
                continue

            timings["detect_ms"] = per_image_detect_ms
            timings["detect_batch_ms_total"] = detect_batch_ms
            fast_faces_ds = eligible_fast_faces_by_index.get(idx, [])
            fast_faces = scale_faces_to_original(
                fast_faces_ds, original.shape, downscaled.shape
            )
            result = self._prepare_face_from_detection(
                original, image_quality, fast_faces, timings
            )
            _observe_timings(result["timings"])
            results.append(result)

        return results

    def process(self, image_bytes: bytes) -> Dict[str, Any]:
        prepared = self.prepare_face_input(image_bytes)
        if prepared["status"] != "ok":
            return prepared

        assert self.encoder is not None, "encoder not initialized"

        face_input = prepared.pop("face_input")
        timings = prepared["timings"]
        bbox_source = prepared["bbox_source"]
        bbox = prepared["bbox"]
        landmarks = prepared.get("landmarks")
        liveness_passed = prepared.get("liveness_passed")
        liveness_score = prepared.get("liveness_score")
        liveness_spoof_score = prepared.get("liveness_spoof_score")
        quality_details = prepared.get("quality_details")

        stage_timings = StageTimings()
        t0 = now_perf_ns()
        embeddings = self.encoder.encode_batch([face_input])
        if len(embeddings) == 0:
            raise RuntimeError("Batch encoder returned no embeddings")
        embedding = embeddings[0]
        stage_timings.finish("encode_ms", t0)
        timings["encode_ms"] = stage_timings.values["encode_ms"]

        if "liveness_ms" in timings and timings["liveness_ms"] > timings["encode_ms"]:
            logger.warning(
                "liveness_slower_than_encode liveness_ms=%.3f encode_ms=%.3f bbox_source=%s",
                timings["liveness_ms"],
                timings["encode_ms"],
                bbox_source,
            )
        if timings.get("liveness_ms", 0.0) > 50:
            logger.warning("slow_liveness_ms=%.2f", timings["liveness_ms"])

        finalized_timings = _finalize_timings(timings)
        _observe_timings(finalized_timings)
        logger.info("bbox_source=%s", bbox_source)

        return {
            "status": "ok",
            "embedding": embedding,
            "bbox": bbox,
            "landmarks": landmarks,
            "liveness_passed": liveness_passed,
            "liveness_score": liveness_score,
            "liveness_spoof_score": liveness_spoof_score,
            "bbox_source": bbox_source,
            "quality_details": quality_details,
            "timings": finalized_timings,
        }
