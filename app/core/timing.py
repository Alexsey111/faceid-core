from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter_ns, time_ns
from typing import Dict


def now_epoch_ns() -> int:
    return time_ns()


def now_perf_ns() -> int:
    return perf_counter_ns()


def elapsed_ms(start_ns: int, end_ns: int | None = None) -> float:
    end = perf_counter_ns() if end_ns is None else end_ns
    return (end - start_ns) / 1_000_000


@dataclass
class StageTimings:
    values: Dict[str, float] = field(default_factory=dict)

    def finish(self, stage: str, start_ns: int) -> float:
        value = elapsed_ms(start_ns)
        self.values[stage] = value
        return value

    def set(self, stage: str, value_ms: float) -> None:
        self.values[stage] = float(value_ms)
