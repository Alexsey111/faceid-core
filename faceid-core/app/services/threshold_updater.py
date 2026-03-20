from app.core.config import settings


def update_thresholds(result: dict):
    """
    Update runtime thresholds after calibration.
    """
    settings.HIGH_THRESHOLD = result.get("high_threshold", settings.HIGH_THRESHOLD)
    settings.LOW_THRESHOLD = result.get("low_threshold", settings.LOW_THRESHOLD)
