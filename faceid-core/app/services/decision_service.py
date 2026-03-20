# decision_service.py - Централизованная логика принятия решения

from typing import Tuple

from app.core.config import settings


class DecisionService:
    """
    Централизованная логика принятия решения о верификации.
    """

    HIGH_THRESHOLD = settings.HIGH_THRESHOLD
    LOW_THRESHOLD = settings.LOW_THRESHOLD
    MARGIN_THRESHOLD = getattr(settings, "MARGIN_THRESHOLD", 0.1)

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

        effective_threshold = DecisionService.HIGH_THRESHOLD

        # если margin плохой → повышаем требования
        if margin < DecisionService.MARGIN_THRESHOLD:
            effective_threshold += 0.05

        if similarity >= effective_threshold and margin >= DecisionService.MARGIN_THRESHOLD:
            status = "match"
            confidence = "high"
        elif similarity >= DecisionService.LOW_THRESHOLD:
            status = "low_confidence"
            confidence = "medium"

        # Liveness is only a signal, doesn't override status
        # (handled in verification_service via liveness_passed field)

        confidence_score = similarity

        # усиливаем/ослабляем confidence через margin
        confidence_score += margin * 0.5

        # clamp
        confidence_score = max(0.0, min(1.0, confidence_score))

        return status, confidence

    @staticmethod
    def compute_confidence_score(similarity: float, margin: float) -> float:
        score = similarity + margin * 0.5
        return max(0.0, min(1.0, score))

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
