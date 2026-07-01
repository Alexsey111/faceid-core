from collections import defaultdict
import time


class Metrics:
    def __init__(self):
        self.counters = defaultdict(int)
        self.timings = defaultdict(list)

    def inc(self, name: str):
        self.counters[name] += 1

    def observe(self, name: str, value: float):
        self.timings[name].append(value)

    def get_stats(self):
        stats = {}

        for k, v in self.timings.items():
            if not v:
                continue

            sorted_v = sorted(v)
            n = len(sorted_v)

            stats[k] = {
                "count": n,
                "p50": sorted_v[int(n * 0.5)],
                "p95": sorted_v[int(n * 0.95)],
                "max": max(sorted_v),
            }

        return {
            "counters": dict(self.counters),
            "timings": stats
        }


metrics = Metrics()