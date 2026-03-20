import numpy as np

from app.services.calibration_service import CalibrationService


def test_calibration_finds_reasonable_thresholds():
    # имитируем данные
    same = np.random.normal(0.75, 0.05, 100)
    diff = np.random.normal(0.25, 0.05, 100)

    scores = np.concatenate([same, diff])
    labels = np.array([1]*100 + [0]*100)

    result = CalibrationService.find_best_thresholds(
        scores.tolist(),
        labels.tolist()
    )

    assert 0.5 < result["high_threshold"] < 0.9
    assert 0.2 < result["low_threshold"] < result["high_threshold"]


def test_metrics_monotonicity():
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 1, 0, 0]

    m1 = CalibrationService.compute_metrics(scores, labels, 0.3)
    m2 = CalibrationService.compute_metrics(scores, labels, 0.7)

    assert m2["precision"] >= m1["precision"]