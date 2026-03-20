from app.core.metrics import Metrics


def test_metrics_basic():
    m = Metrics()

    m.inc("test_counter")
    m.observe("latency", 0.1)
    m.observe("latency", 0.2)

    stats = m.get_stats()

    assert stats["counters"]["test_counter"] == 1
    assert stats["timings"]["latency"]["count"] == 2