# liveness_service.py - Сервис объединения сигналов liveness

from typing import Dict, Any

from app.core.config import settings


class LivenessService:
    """
    Объединяет multiple liveness signals в итоговый score и risk.
    """

    @staticmethod
    def fuse(signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Объединяет сигналы в итоговый score и риск.

        Args:
            signals: Dict с ключами "passive", "texture", "blur"

        Returns:
            Dict с "score" (float 0-1) и "risk" (low/medium/high)
        """
        passive = signals.get("passive") or 0.0
        texture = signals.get("texture") or 0.0
        blur = signals.get("blur") or 0.0

        # Веса (эмпирически)
        # Если passive модель недоступна, перераспределяем веса
        if signals.get("passive") is not None:
            score = (
                0.6 * passive +
                0.25 * texture +
                0.15 * blur
            )
        else:
            # Без passive модели - полагаемся на texture и blur
            score = (
                0.7 * texture +
                0.3 * blur
            )

        # Risk classification
        if score >= 0.8:
            risk = "low"
        elif score >= 0.6:
            risk = "medium"
        else:
            risk = "high"

        return {
            "score": float(score),
            "risk": risk
        }

    @staticmethod
    def is_passed(signals: Dict[str, Any]) -> bool:
        """
        Проверяет, прошёл ли liveness check.

        Args:
            signals: Dict с сигналами liveness

        Returns:
            bool: True если score >= LIVENESS_THRESHOLD
        """
        result = LivenessService.fuse(signals)
        return result["score"] >= settings.LIVENESS_THRESHOLD