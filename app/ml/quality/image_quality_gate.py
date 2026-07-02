from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.core.config import settings


@dataclass(slots=True)
class QualityCheckResult:
    passed: bool
    reason: str | None
    details: dict[str, Any]


class ImageQualityGate:
    """
    Lightweight quality gate with configurable thresholds and modes.

    Modes:
    - hard: reject on threshold violations
    - soft: never reject, but return warning details
    - off: always pass, with minimal details
    """

    def __init__(self) -> None:
        self.mode = settings.QUALITY_GATE_MODE
        self.pose_mode = settings.POSE_QUALITY_MODE
        self.lighting_mode = settings.QUALITY_LIGHTING_MODE

        self.min_image_side = int(settings.QUALITY_MIN_IMAGE_SIDE)
        self.min_blur_score = float(settings.QUALITY_MIN_BLUR_SCORE)
        self.min_brightness = float(settings.QUALITY_MIN_BRIGHTNESS)
        self.max_brightness = float(settings.QUALITY_MAX_BRIGHTNESS)
        self.min_contrast = float(settings.QUALITY_MIN_CONTRAST)

        self.min_face_side = int(settings.QUALITY_MIN_FACE_SIDE)
        self.max_eye_line_diff_ratio = float(settings.QUALITY_MAX_EYE_LINE_DIFF_RATIO)
        self.max_nose_offset_ratio = float(settings.QUALITY_MAX_NOSE_OFFSET_RATIO)

        # Lighting (capture-качество): свой режим, независимый от QUALITY_GATE_MODE.
        self.min_lighting_uniformity = float(settings.QUALITY_MIN_LIGHTING_UNIFORMITY)
        self.max_shadow_asymmetry = float(settings.QUALITY_MAX_SHADOW_ASYMMETRY)

        # Окклюзия (маска/очки) — режимов hard/soft/off НЕТ: retry всегда при
        # срабатывании. Тумблеры включают/выключают детекцию каждого типа отдельно.
        self.mask_detect_enabled = bool(settings.QUALITY_MASK_DETECT_ENABLED)
        self.min_lower_face_skin_frac = float(settings.QUALITY_MIN_LOWER_FACE_SKIN_FRAC)
        self.glasses_detect_enabled = bool(settings.QUALITY_GLASSES_DETECT_ENABLED)
        self.max_eye_edge_density = float(settings.QUALITY_MAX_EYE_EDGE_DENSITY)

    def _wrap_result(
        self,
        *,
        passed: bool,
        reason: str | None,
        details: dict[str, Any],
        stage: str,
        mode: str | None = None,
    ) -> QualityCheckResult:
        # mode: какой режим применить для смягчения. По умолчанию — общий
        # QUALITY_GATE_MODE. Lighting передаёт свой QUALITY_LIGHTING_MODE, чтобы
        # soft/hard lighting были независимы от общего gate. Окклюзия сюда НЕ идёт
        # (она блокирует всегда — отдельный возврат в evaluate_detection).
        eff_mode = self.mode if mode is None else mode
        base_details = {
            **details,
            "quality_gate_mode": eff_mode,
            "quality_stage": stage,
            "quality_warning": reason,
            "quality_hard_reject": bool(eff_mode == "hard" and not passed),
        }

        if eff_mode == "off":
            return QualityCheckResult(passed=True, reason=None, details=base_details)

        if eff_mode == "soft":
            return QualityCheckResult(passed=True, reason=None, details=base_details)

        return QualityCheckResult(passed=passed, reason=reason, details=base_details)

    def evaluate_image(self, image: np.ndarray) -> QualityCheckResult:
        if image is None or image.size == 0:
            return self._wrap_result(
                passed=False,
                reason="invalid_image",
                details={},
                stage="image",
            )

        height, width = image.shape[:2]
        min_side = min(height, width)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        contrast = float(gray.std())

        details = {
            "image_width": int(width),
            "image_height": int(height),
            "min_image_side": int(min_side),
            "blur_score": round(blur_score, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
        }

        if min_side < self.min_image_side:
            return self._wrap_result(
                passed=False,
                reason="image_too_small",
                details=details,
                stage="image",
            )

        if blur_score < self.min_blur_score:
            return self._wrap_result(
                passed=False,
                reason="image_blurry",
                details=details,
                stage="image",
            )

        if brightness < self.min_brightness:
            return self._wrap_result(
                passed=False,
                reason="image_too_dark",
                details=details,
                stage="image",
            )

        if brightness > self.max_brightness:
            return self._wrap_result(
                passed=False,
                reason="image_too_bright",
                details=details,
                stage="image",
            )

        if contrast < self.min_contrast:
            return self._wrap_result(
                passed=False,
                reason="low_contrast",
                details=details,
                stage="image",
            )

        return self._wrap_result(
            passed=True,
            reason=None,
            details=details,
            stage="image",
        )

    def evaluate_detection(
        self,
        bbox: list[int] | tuple[int, int, int, int],
        landmarks: Any | None,
        image: np.ndarray | None = None,
    ) -> QualityCheckResult:
        x1, y1, x2, y2 = map(int, bbox)
        face_w = max(0, x2 - x1)
        face_h = max(0, y2 - y1)
        min_face_side = min(face_w, face_h)

        details: dict[str, Any] = {
            "face_width": int(face_w),
            "face_height": int(face_h),
            "min_face_side": int(min_face_side),
            "bbox_x1": int(x1),
            "bbox_y1": int(y1),
            "bbox_x2": int(x2),
            "bbox_y2": int(y2),
        }

        if min_face_side < self.min_face_side:
            return self._wrap_result(
                passed=False,
                reason="face_too_small",
                details=details,
                stage="face",
            )

        pose_check = self._check_pose(landmarks, face_w, face_h)
        details.update(pose_check["details"])

        if not pose_check["passed"]:
            return self._wrap_result(
                passed=False,
                reason=pose_check["reason"],
                details=details,
                stage="face",
            )

        # Новые проверки (нужен кроп лица). Без image — пропускаются (backward-compat:
        # существующие вызовы evaluate_detection(bbox, landmarks) работают как раньше).
        if image is not None:
            face_crop, origin = self._face_crop(image, bbox)
            if face_crop is not None and face_crop.size > 0:
                pts_crop = self._landmarks_in_crop(landmarks, origin, face_crop.shape[:2])

                # Окклюзия (маска/очки) — блокирующий retry, независим от режимов.
                # Проверяем ДО lighting: masked-лицо → «снимите маску» полезнее, чем
                # «улучшите свет» (снятие маски часто и свет выправит).
                occ_flags = self._check_occlusion(face_crop, pts_crop)
                details["occlusion_flags"] = occ_flags
                if occ_flags["mask_detected"] or occ_flags["glasses_detected"]:
                    # Не через _wrap_result: retry всегда, soft/hard/off не действуют.
                    occ_details = {
                        **details,
                        "quality_gate_mode": self.mode,
                        "quality_stage": "face",
                        "quality_warning": "remove_occlusion",
                        "quality_hard_reject": True,
                    }
                    return QualityCheckResult(
                        passed=False,
                        reason="remove_occlusion",
                        details=occ_details,
                    )

                # Lighting (capture-качество) — свой режим QUALITY_LIGHTING_MODE.
                nose_x = float(pts_crop[2, 0]) if pts_crop is not None else None
                lighting_check = self._check_lighting(face_crop, nose_x)
                details.update(lighting_check["details"])
                if not lighting_check["passed"]:
                    return self._wrap_result(
                        passed=False,
                        reason=lighting_check["reason"],
                        details=details,
                        stage="face",
                        mode=self.lighting_mode,
                    )

        return self._wrap_result(
            passed=True,
            reason=None,
            details=details,
            stage="face",
        )

    def _check_pose(
        self,
        landmarks: Any | None,
        face_w: int,
        face_h: int,
    ) -> dict[str, Any]:
        # Режим pose-check независим от общего QUALITY_GATE_MODE:
        #   off  — проверка полностью отключена (passed=True, skipped)
        #   soft — bad_pose помечается как warning (passed=True), запрос не
        #          отбрасывается, чтобы не поднимать False Reject на повёрнутых
        #          лицах; сигнал остаётся в details.pose_warning
        #   hard — bad_pose отбрасывает запрос (passed=False)
        pose_mode = self.pose_mode

        if pose_mode == "off":
            return {
                "passed": True,
                "reason": None,
                "details": {"pose_check_skipped": True, "pose_check_mode": "off"},
            }

        if landmarks is None:
            return {
                "passed": True,
                "reason": None,
                "details": {"pose_check_skipped": True, "pose_check_mode": pose_mode},
            }

        try:
            pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
        except Exception:
            return {
                "passed": True,
                "reason": None,
                "details": {"pose_check_skipped": True, "pose_check_mode": pose_mode},
            }

        if pts.shape[0] < 5 or face_w <= 0 or face_h <= 0:
            return {
                "passed": True,
                "reason": None,
                "details": {"pose_check_skipped": True, "pose_check_mode": pose_mode},
            }

        left_eye = pts[0]
        right_eye = pts[1]
        nose = pts[2]

        eye_line_diff_ratio = abs(float(left_eye[1] - right_eye[1])) / float(max(face_h, 1))
        eye_center_x = float((left_eye[0] + right_eye[0]) / 2.0)
        nose_offset_ratio = abs(float(nose[0] - eye_center_x)) / float(max(face_w, 1))

        details = {
            "pose_check_skipped": False,
            "pose_check_mode": pose_mode,
            "eye_line_diff_ratio": round(eye_line_diff_ratio, 4),
            "nose_offset_ratio": round(nose_offset_ratio, 4),
        }

        exceeded = (
            eye_line_diff_ratio > self.max_eye_line_diff_ratio
            or nose_offset_ratio > self.max_nose_offset_ratio
        )

        if not exceeded:
            return {"passed": True, "reason": None, "details": details}

        # превышение порогов pose
        if pose_mode == "soft":
            # warning-only: не отбрасываем, но сигнализируем
            details["pose_warning"] = "bad_pose"
            return {"passed": True, "reason": "bad_pose", "details": details}

        # hard
        return {"passed": False, "reason": "bad_pose", "details": details}

    # ------------------------------------------------------------------
    # Lighting / shadow / occlusion (новые проверки п.5 аудита)
    # ------------------------------------------------------------------

    def _face_crop(
        self,
        image: np.ndarray,
        bbox: list[int] | tuple[int, int, int, int],
    ) -> tuple[np.ndarray | None, tuple[int, int]]:
        """Вырезать кроп лица по bbox с clamping к границам кадра.

        Возвращает (crop, origin) — origin = (x1, y1) кропа в координатах исходного
        изображения (нужно для перевода landmarks в систему координат кропа).
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1c = max(0, min(x1, w))
        y1c = max(0, min(y1, h))
        x2c = max(0, min(x2, w))
        y2c = max(0, min(y2, h))
        if x2c <= x1c or y2c <= y1c:
            return None, (0, 0)
        crop = image[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            return None, (x1c, y1c)
        return np.ascontiguousarray(crop), (x1c, y1c)

    @staticmethod
    def _landmarks_in_crop(
        landmarks: Any | None,
        origin: tuple[int, int],
        crop_shape: tuple[int, int],
    ) -> np.ndarray | None:
        """Перевести 5-pt landmarks в координаты кропа. None при отсутствии/ошибке."""
        if landmarks is None:
            return None
        try:
            pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
        except Exception:
            return None
        if pts.shape[0] < 5:
            return None
        ox, oy = float(origin[0]), float(origin[1])
        pts = pts.copy()
        pts[:, 0] -= ox
        pts[:, 1] -= oy
        return pts

    def _check_lighting(
        self,
        face_crop: np.ndarray,
        nose_x: float | None,
    ) -> dict[str, Any]:
        """Равномерность освещения (сетка 3×3) + жёсткая тень (L/R асимметрия).

        Режим QUALITY_LIGHTING_MODE: off → пропустить; soft → warning-only
        (passed=True, lighting_warning в details); hard → passed=False.
        """
        mode = self.lighting_mode
        if mode == "off":
            return {
                "passed": True,
                "reason": None,
                "details": {"lighting_check_skipped": True, "lighting_check_mode": "off"},
            }

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if h < 6 or w < 6:
            return {
                "passed": True,
                "reason": None,
                "details": {
                    "lighting_check_skipped": True,
                    "lighting_check_mode": mode,
                },
            }

        # Равномерность: min_cell_mean / max_cell_mean по сетке 3×3.
        gh, gw = h // 3, w // 3
        cell_means: list[float] = []
        for i in range(3):
            for j in range(3):
                cell = gray[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw]
                if cell.size > 0:
                    cell_means.append(float(cell.mean()))
        if cell_means:
            mx = max(cell_means)
            mn = min(cell_means)
            uniformity = float(mn / mx) if mx > 0 else 1.0
        else:
            uniformity = 1.0

        # Тень: split по вертикали через nose_x (если есть), иначе по центру.
        overall = float(gray.mean())
        if nose_x is not None:
            nx = int(max(1, min(int(nose_x), w - 1)))
        else:
            nx = w // 2
        mean_left = float(gray[:, :nx].mean()) if nx > 0 else overall
        mean_right = float(gray[:, nx:].mean()) if nx < w else overall
        shadow_asymmetry = (
            float(abs(mean_left - mean_right) / overall) if overall > 0 else 0.0
        )

        details = {
            "lighting_check_skipped": False,
            "lighting_check_mode": mode,
            "lighting_uniformity": round(uniformity, 3),
            "shadow_asymmetry": round(shadow_asymmetry, 3),
        }

        # Тень приоритетнее (более специфичный сигнал бокового света).
        violated: str | None = None
        if shadow_asymmetry > self.max_shadow_asymmetry:
            violated = "hard_shadow"
        elif uniformity < self.min_lighting_uniformity:
            violated = "bad_lighting"

        if violated is None:
            return {"passed": True, "reason": None, "details": details}

        if mode == "soft":
            details["lighting_warning"] = violated
            return {"passed": True, "reason": violated, "details": details}

        # hard
        return {"passed": False, "reason": violated, "details": details}

    def _check_occlusion(
        self,
        face_crop: np.ndarray,
        pts_crop: np.ndarray | None,
    ) -> dict[str, Any]:
        """Детекция маски (skin-tone в нижней зоне) и очков (edge-density в зоне глаз).

        Метаданные + блокирующий сигнал: returns флаги; решение об retry принимает
        evaluate_detection по mask_detected/glasses_detected. Режимов hard/soft/off
        НЕТ — окклюзия всегда требует пере-съёмки (тумблеры *_DETECT_ENABLED отключают
        детекцию типа, но не смягчают до warning).
        """
        flags: dict[str, Any] = {
            "mask_detected": False,
            "glasses_detected": False,
            "lower_face_skin_frac": None,
            "eye_edge_density": None,
        }
        h, w = face_crop.shape[:2]
        if pts_crop is None:
            return flags

        nose = pts_crop[2]
        mouth_l = pts_crop[3]
        mouth_r = pts_crop[4]
        left_eye = pts_crop[0]
        right_eye = pts_crop[1]

        # --- Маска: skin-tone фракция в нижней зоне лица (нос → подбородок). ---
        if self.mask_detect_enabled:
            mouth_dx = float(abs(mouth_r[0] - mouth_l[0]))
            x_l = int(max(0, min(mouth_l[0], mouth_r[0]) - 0.15 * mouth_dx))
            x_r = int(min(w, max(mouth_l[0], mouth_r[0]) + 0.15 * mouth_dx))
            y_top = int(max(0, min(nose[1], h)))
            y_bot = int(min(h, max(y_top + 1, int(h * 0.95))))
            if x_r > x_l and y_bot > y_top:
                region = face_crop[y_top:y_bot, x_l:x_r]
                if region.size > 0:
                    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                    h_ch = hsv[:, :, 0]
                    s_ch = hsv[:, :, 1]
                    v_ch = hsv[:, :, 2]
                    skin = (
                        ((h_ch <= 25) | (h_ch >= 165))
                        & (s_ch >= 25)
                        & (s_ch <= 170)
                        & (v_ch >= 40)
                    )
                    frac = float(skin.sum()) / float(region.shape[0] * region.shape[1])
                    flags["lower_face_skin_frac"] = round(frac, 3)
                    if frac < self.min_lower_face_skin_frac:
                        flags["mask_detected"] = True

        # --- Очки: энергия градиента (Sobel) в квадратных патчах вокруг глаз. ---
        # Canny с NMS даёт нестабильную долю edge-пикселей; Sobel-magnitude mean
        # монотоннее и управляемее. Гладкая кожа → ~0, оправа/линзы → высокий
        # градиент. Порог нормирован на 255 (значения могут >1 для 0↔255 контраста).
        if self.glasses_detect_enabled:
            eye_dist = float(abs(right_eye[0] - left_eye[0]))
            side = max(8, int(eye_dist / 3.0))
            densities: list[float] = []
            for eye in (left_eye, right_eye):
                ex, ey = int(eye[0]), int(eye[1])
                y0, y1 = max(0, ey - side), min(h, ey + side)
                x0, x1 = max(0, ex - side), min(w, ex + side)
                if y1 > y0 and x1 > x0:
                    patch = face_crop[y0:y1, x0:x1]
                    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    gx = cv2.Sobel(patch_gray, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(patch_gray, cv2.CV_32F, 0, 1, ksize=3)
                    mag = cv2.magnitude(gx, gy)
                    densities.append(float(mag.mean()) / 255.0)
            if densities:
                ed = float(np.mean(densities))
                flags["eye_edge_density"] = round(ed, 3)
                if ed > self.max_eye_edge_density:
                    flags["glasses_detected"] = True

        return flags
