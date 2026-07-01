from typing import List
from sqlalchemy import select

from app.services.calibration_service import CalibrationService
from app.models.verification_log import VerificationLog
from app.monitoring.db_metrics import timed_db_call


class AutoCalibrationService:

    def __init__(self, db):
        self.db = db

    async def load_data(self) -> tuple[List[float], List[int]]:
        """
        Загружаем данные из verification_logs

        labels:
        1 = true match
        0 = false match
        """

        result = await timed_db_call(
            self.db.execute(
                select(
                    VerificationLog.similarity,
                    VerificationLog.is_genuine
                )
            ),
            "auto_calibration.load_data",
        )

        rows = result.fetchall()

        scores = []
        labels = []

        for sim, is_genuine in rows:
            if sim is None or is_genuine is None:
                continue

            scores.append(float(sim))
            labels.append(1 if is_genuine else 0)

        return scores, labels

    async def calibrate(self):
        scores, labels = await self.load_data()

        if len(scores) < 50:
            # fallback → всё равно считаем, но помечаем как "low confidence"
            result = CalibrationService.find_best_thresholds(scores, labels)
            result["low_data"] = True
            return result

        result = CalibrationService.find_best_thresholds(scores, labels)

        return result
