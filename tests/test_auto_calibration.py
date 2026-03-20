import pytest
from typing import cast

from app.services.auto_calibration_service import AutoCalibrationService


class FakeDB:
    async def execute(self, query):
        class Result:
            def fetchall(self):
                # симулируем данные
                return [
                    (0.8, True),
                    (0.75, True),
                    (0.2, False),
                    (0.3, False),
                    (0.7, True),
                    (0.25, False),
                ]
        return Result()


@pytest.mark.asyncio
async def test_auto_calibration_runs():
    service = AutoCalibrationService(FakeDB())

    result = await service.calibrate()

    assert result is not None
    assert "high_threshold" in result
    assert "low_threshold" in result
