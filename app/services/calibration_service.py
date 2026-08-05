import numpy as np
from typing import List, Dict


class CalibrationService:
    """
    Подбор оптимальных threshold'ов для face verification.
    """

    @staticmethod
    def compute_metrics(scores: List[float], labels: List[int], threshold: float):
        """
        labels: 1 = same person, 0 = different
        """
        tp = fp = tn = fn = 0

        for score, label in zip(scores, labels):
            pred = 1 if score >= threshold else 0

            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 0:
                tn += 1
            else:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0

        return {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

    @staticmethod
    def find_best_thresholds(
        scores: List[float],
        labels: List[int],
    ) -> Dict:
        """
        Подбираем:
        - high_threshold → максимум precision
        - low_threshold → максимум recall
        """

        thresholds = np.linspace(0.0, 1.0, 100)

        MIN_PRECISION_HIGH = 0.95
        MIN_PRECISION_LOW = 0.5

        best_high = None
        best_low = None

        best_recall = 0

        metrics = []

        for t in thresholds:
            m = CalibrationService.compute_metrics(scores, labels, t)
            metrics.append(m)

            precision = m["precision"]
            recall = m["recall"]

            if precision >= MIN_PRECISION_HIGH:
                best_high = t

            if precision >= MIN_PRECISION_LOW and recall >= best_recall:
                best_recall = recall
                best_low = t

        if best_high is None:
            best_high = 0.5

        if best_low is None:
            best_low = 0.3

        return {
            "high_threshold": best_high,
            "low_threshold": best_low,
            "metrics": metrics,
        }
