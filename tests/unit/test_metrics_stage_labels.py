from __future__ import annotations

from app.monitoring import metrics


def test_pipeline_stage_metrics_are_low_cardinality():
    assert tuple(metrics.ASYNC_STAGE_LATENCY_MS._labelnames) == ("stage",)
    assert tuple(metrics.PIPELINE_STAGE_LATENCY_MS._labelnames) == ("stage",)
    assert tuple(metrics.ASYNC_STAGE_FAILURES_TOTAL._labelnames) == ("stage", "reason")

    for labelnames in (
        metrics.ASYNC_STAGE_LATENCY_MS._labelnames,
        metrics.PIPELINE_STAGE_LATENCY_MS._labelnames,
        metrics.ASYNC_STAGE_FAILURES_TOTAL._labelnames,
    ):
        assert "job_id" not in labelnames
        assert "user_id" not in labelnames
