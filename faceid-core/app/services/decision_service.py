# decision_service.py - Централизованная логика принятия решения

from typing import Tuple

from app.core.config import settings


class DecisionService:
    """
    Централизованная логика принятия решения о верификации.
    """

    HIGH_THRESHOLD = 0.55
    LOW_THRESHOLD = 0.45
    MARGIN_THRESHOLD = 0.1

    @staticmethod
    def decide(
        similarity: float,
        margin: float,
        liveness_score: float | None = None
    ) -> Tuple[str, str]:
        """
        Принимает решение на основе similarity и margin.

        Args:
            similarity: Косинусное сходство с лучшим совпадением
            margin: Разница между top1 и top2 (защита от похожих лиц)
            liveness_score: Оценка liveness (опционально)

        Returns:
            Tuple[str, str]: (status, confidence)
                status: match | low_confidence | no_match | spoof
                confidence: high | medium | low
        """
        # Base decision
        status = "no_match"
        confidence = "low"

        if similarity >= DecisionService.HIGH_THRESHOLD and margin >= DecisionService.MARGIN_THRESHOLD:
            status = "match"
            confidence = "high"
        elif similarity >= DecisionService.LOW_THRESHOLD:
            status = "low_confidence"
            confidence = "medium"

        # Liveness is only a signal, doesn't override status
        # (handled in verification_service via liveness_passed field)

        return status, confidence

    @staticmethod
    def check_liveness(liveness_score: float) -> bool:
        """
        Проверка liveness.

        Args:
            liveness_score: Оценка liveness модели

        Returns:
            bool: True если прошёл проверку
        """
        return liveness_score >= settings.LIVENESS_THRESHOLD
