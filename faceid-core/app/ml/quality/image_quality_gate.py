from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class QualityCheckResult:
    passed: bool
    reason: str | None
    details: dict[str, Any]


class ImageQualityGate:
    """
    Лёгкий quality gate без тяжёлых ML-операций.

    Этапы:
    1. Pre-detect checks:
       - min image size
       - blur
       - brightness
       - contrast

    2. Post-detect checks:
       - min face size
       - basic pose check (если есть landmarks)
    """

    MIN_IMAGE_SIDE = 160

    # Чем меньше variance of Laplacian, тем сильнее blur
    MIN_BLUR_SCORE = 45.0

    # Средняя яркость grayscale
    MIN_BRIGHTNESS = 35.0
    MAX_BRIGHTNESS = 225.0

    # std grayscale
    MIN_CONTRAST = 18.0

    # Минимальный размер лица по короткой стороне bbox
    MIN_FACE_SIDE = 72

    # Грубые эвристики по pose
    MAX_EYE_LINE_DIFF_RATIO = 0.12
    MAX_NOSE_OFFSET_RATIO = 0.18

    def evaluate_image(self, image: np.ndarray) -> QualityCheckResult:
        """
        Проверка качества полного изображения до detector.
        """
        if image is None or image.size == 0:
            return QualityCheckResult(
                passed=False,
                reason="invalid_image",
                details={},
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

        if min_side < self.MIN_IMAGE_SIDE:
            return QualityCheckResult(
                passed=False,
                reason="image_too_small",
                details=details,
            )

        if blur_score < self.MIN_BLUR_SCORE:
            return QualityCheckResult(
                passed=False,
                reason="image_blurry",
                details=details,
            )

        if brightness < self.MIN_BRIGHTNESS:
            return QualityCheckResult(
                passed=False,
                reason="image_too_dark",
                details=details,
            )

        if brightness > self.MAX_BRIGHTNESS:
            return QualityCheckResult(
                passed=False,
                reason="image_too_bright",
                details=details,
            )

        if contrast < self.MIN_CONTRAST:
            return QualityCheckResult(
                passed=False,
                reason="low_contrast",
                details=details,
            )

        return QualityCheckResult(
            passed=True,
            reason=None,
            details=details,
        )

    def evaluate_detection(
        self,
        bbox: list[int] | tuple[int, int, int, int],
        landmarks: Any | None,
    ) -> QualityCheckResult:
        """
        Проверка качества уже найденного лица.
        """
        x1, y1, x2, y2 = map(int, bbox)
        face_w = max(0, x2 - x1)
        face_h = max(0, y2 - y1)
        min_face_side = min(face_w, face_h)

        details: dict[str, Any] = {
            "face_width": int(face_w),
            "face_height": int(face_h),
            "min_face_side": int(min_face_side),
        }

        if min_face_side < self.MIN_FACE_SIDE:
            return QualityCheckResult(
                passed=False,
                reason="face_too_small",
                details=details,
            )

        pose_check = self._check_pose(landmarks, face_w, face_h)
        details.update(pose_check["details"])

        if not pose_check["passed"]:
            return QualityCheckResult(
                passed=False,
                reason=pose_check["reason"],
                details=details,
            )

        return QualityCheckResult(
            passed=True,
            reason=None,
            details=details,
        )

    def _check_pose(
        self,
        landmarks: Any | None,
        face_w: int,
        face_h: int,
    ) -> dict[str, Any]:
        """
        Базовая pose-проверка.
        Если landmarks нет, не режем кадр по pose.
        """
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

        if eye_line_diff_ratio > self.MAX_EYE_LINE_DIFF_RATIO:
            return {
                "passed": False,
                "reason": "bad_pose",
                "details": details,
            }

        if nose_offset_ratio > self.MAX_NOSE_OFFSET_RATIO:
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