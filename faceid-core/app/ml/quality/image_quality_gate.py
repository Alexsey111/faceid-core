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

        self.min_image_side = int(settings.QUALITY_MIN_IMAGE_SIDE)
        self.min_blur_score = float(settings.QUALITY_MIN_BLUR_SCORE)
        self.min_brightness = float(settings.QUALITY_MIN_BRIGHTNESS)
        self.max_brightness = float(settings.QUALITY_MAX_BRIGHTNESS)
        self.min_contrast = float(settings.QUALITY_MIN_CONTRAST)

        self.min_face_side = int(settings.QUALITY_MIN_FACE_SIDE)
        self.max_eye_line_diff_ratio = float(settings.QUALITY_MAX_EYE_LINE_DIFF_RATIO)
        self.max_nose_offset_ratio = float(settings.QUALITY_MAX_NOSE_OFFSET_RATIO)

    def _wrap_result(
        self,
        *,
        passed: bool,
        reason: str | None,
        details: dict[str, Any],
        stage: str,
    ) -> QualityCheckResult:
        base_details = {
            **details,
            "quality_gate_mode": self.mode,
            "quality_stage": stage,
            "quality_warning": reason,
            "quality_hard_reject": bool(self.mode == "hard" and not passed),
        }

        if self.mode == "off":
            return QualityCheckResult(passed=True, reason=None, details=base_details)

        if self.mode == "soft":
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
        if landmarks is None:
            return {
                "passed": True,
                "reason": None,
                "details": {"pose_check_skipped": True},
            }

        try:
            pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
        except Exception:
            return {
                "passed": True,
                "reason": None,
                "details": {"pose_check_skipped": True},
            }

        if pts.shape[0] < 5 or face_w <= 0 or face_h <= 0:
            return {
                "passed": True,
                "reason": None,
                "details": {"pose_check_skipped": True},
            }

        left_eye = pts[0]
        right_eye = pts[1]
        nose = pts[2]

        eye_line_diff_ratio = abs(float(left_eye[1] - right_eye[1])) / float(max(face_h, 1))
        eye_center_x = float((left_eye[0] + right_eye[0]) / 2.0)
        nose_offset_ratio = abs(float(nose[0] - eye_center_x)) / float(max(face_w, 1))

        details = {
            "pose_check_skipped": False,
            "eye_line_diff_ratio": round(eye_line_diff_ratio, 4),
            "nose_offset_ratio": round(nose_offset_ratio, 4),
        }

        if eye_line_diff_ratio > self.max_eye_line_diff_ratio:
            return {
                "passed": False,
                "reason": "bad_pose",
                "details": details,
            }

        if nose_offset_ratio > self.max_nose_offset_ratio:
            return {
                "passed": False,
                "reason": "bad_pose",
                "details": details,
            }

        return {
            "passed": True,
            "reason": None,
            "details": details,
        }
