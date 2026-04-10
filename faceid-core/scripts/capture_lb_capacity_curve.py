from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
K6_SCRIPT = ROOT / "load_test_async.js"
DATA_FILE = ROOT / "tests" / "data" / "person1_small.b64.txt"
SUMMARY_DIR = ROOT / "benchmarks" / "lb_capacity_curve"
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8080")
DURATION = os.environ.get("DURATION", "60s")
PRE_ALLOCATED_VUS = os.environ.get("PRE_ALLOCATED_VUS", "64")
MAX_VUS = os.environ.get("MAX_VUS", "128")
DRAIN_TIMEOUT_S = int(os.environ.get("DRAIN_TIMEOUT_S", "180"))
K6_WAIT_TIMEOUT_MS = int(os.environ.get("WAIT_TIMEOUT_MS", "30000"))
if K6_WAIT_TIMEOUT_MS > 30000:
    raise SystemExit(
        f"WAIT_TIMEOUT_MS={K6_WAIT_TIMEOUT_MS} is invalid for /jobs/{{id}}/wait; max supported is 30000"
    )
RATES = [int(x) for x in os.environ.get("RATES", "8,10,12").split(",") if x.strip()]


@dataclass
class SummaryResult:
    completion_failed_pct: float
    completed_eventually_pct: float
    client_iteration_p95_ms: float | None
    client_terminal_e2e_p95_ms: float | None
    iterations: int
    result_fails: int


def fetch_metrics() -> str:
    with urllib.request.urlopen(f"{BASE_URL.rstrip('/')}/metrics", timeout=15) as response:
        return response.read().decode("utf-8", errors="replace")


def save_metrics_snapshot(rate: int, phase: str, metrics_text: str) -> Path:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = SUMMARY_DIR / f"r{rate}_{phase}.metrics"
    snapshot_path.write_text(metrics_text, encoding="utf-8")
    return snapshot_path


def parse_metric_samples(metrics_text: str, metric_name: str) -> tuple[list[tuple[float, float]], float | None]:
    bucket_re = re.compile(
        rf"^{re.escape(metric_name)}_bucket\{{le=\"([^\"]+)\"\}} ([0-9.]+)$",
        re.MULTILINE,
    )
    count_re = re.compile(rf"^{re.escape(metric_name)}_count ([0-9.]+)$", re.MULTILINE)

    buckets: list[tuple[float, float]] = []
    for le, value in bucket_re.findall(metrics_text):
        upper = math.inf if le == "+Inf" else float(le)
        buckets.append((upper, float(value)))
    buckets.sort(key=lambda item: item[0])

    count_match = count_re.search(metrics_text)
    total = float(count_match.group(1)) if count_match else None
    return buckets, total


def parse_gauge(metrics_text: str, metric_name: str) -> float | None:
    re_ = re.compile(rf"^{re.escape(metric_name)}(?:\{{[^}}]*\}})? ([0-9.]+)$", re.MULTILINE)
    matches = re_.findall(metrics_text)
    if not matches:
        return None
    return max(float(value) for value in matches)


def histogram_delta_p95(pre_text: str, post_text: str, metric_name: str) -> float | None:
    pre_buckets, pre_total = parse_metric_samples(pre_text, metric_name)
    post_buckets, post_total = parse_metric_samples(post_text, metric_name)
    if pre_total is None or post_total is None or not pre_buckets or not post_buckets:
        return None

    pre_map = {upper: count for upper, count in pre_buckets}
    post_map = {upper: count for upper, count in post_buckets}
    uppers = sorted(set(pre_map) | set(post_map), key=lambda x: (math.inf if math.isinf(x) else x))

    delta_buckets: list[tuple[float, float]] = []
    for upper in uppers:
        delta_buckets.append((upper, max(0.0, post_map.get(upper, 0.0) - pre_map.get(upper, 0.0))))

    total = max(0.0, post_total - pre_total)
    if total <= 0:
        return None

    target = total * 0.95
    prev_count = 0.0
    prev_upper = 0.0
    for upper, cumulative in delta_buckets:
        if cumulative >= target:
            if math.isinf(upper):
                return prev_upper
            span = cumulative - prev_count
            if span <= 0:
                return upper
            return prev_upper + ((target - prev_count) / span) * (upper - prev_upper)
        prev_count = cumulative
        prev_upper = upper

    return None


def run_k6(rate: int) -> Path:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SUMMARY_DIR / f"r{rate}.json"

    env = os.environ.copy()
    env["BASE_URL"] = BASE_URL
    env["RATE"] = str(rate)
    env["DURATION"] = DURATION
    env["PRE_ALLOCATED_VUS"] = PRE_ALLOCATED_VUS
    env["MAX_VUS"] = MAX_VUS
    env["WAIT_TIMEOUT_MS"] = str(K6_WAIT_TIMEOUT_MS)
    env["IMAGE_FILE"] = str(DATA_FILE)

    cmd = [
        "k6",
        "run",
        "--summary-trend-stats",
        "min,avg,med,p(90),p(95),p(99),max",
        "--summary-export",
        str(summary_path),
        str(K6_SCRIPT),
    ]

    print(f"=== RATE {rate} ===", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if proc.returncode != 0:
        print(f"k6 exit_code={proc.returncode}", file=sys.stderr, flush=True)
    return summary_path


def wait_for_drain() -> None:
    deadline = time.time() + DRAIN_TIMEOUT_S
    stable = 0
    while time.time() < deadline:
        metrics = fetch_metrics()
        queue_pending = parse_gauge(metrics, "faceid_queue_jobs_pending") or 0.0
        inflight = parse_gauge(metrics, "faceid_verify_inflight_current") or 0.0
        http_inflight = parse_gauge(metrics, "faceid_verify_async_http_inflight") or 0.0

        if queue_pending <= 0.0 and inflight <= 0.0 and http_inflight <= 0.0:
            stable += 1
            if stable >= 3:
                return
        else:
            stable = 0

        time.sleep(2)

    print("warning: drain wait timed out; using latest metrics snapshot", file=sys.stderr)


def read_summary(summary_path: Path) -> SummaryResult:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    iterations = int(metrics["iterations"]["count"])
    result_fails = int(metrics.get("result_fails", {}).get("count", 0))
    completion_failed_pct = float(metrics.get("completion_failed", {}).get("value", 0.0) or 0.0) * 100.0
    completed_eventually_pct = float(metrics.get("completed_eventually", {}).get("value", 0.0) or 0.0) * 100.0
    iteration_duration = metrics.get("iteration_duration", {})
    client_iteration_p95_ms = iteration_duration.get("p(95)")
    client_terminal_e2e_metric = metrics.get("client_e2e_ms", {})
    client_terminal_e2e_p95_ms = client_terminal_e2e_metric.get("p(95)")
    return SummaryResult(
        completion_failed_pct=completion_failed_pct,
        completed_eventually_pct=completed_eventually_pct,
        client_iteration_p95_ms=float(client_iteration_p95_ms) if client_iteration_p95_ms is not None else None,
        client_terminal_e2e_p95_ms=
            float(client_terminal_e2e_p95_ms) if client_terminal_e2e_p95_ms is not None else None,
        iterations=iterations,
        result_fails=result_fails,
    )


def main() -> int:
    rows = []
    for rate in RATES:
        pre_metrics = fetch_metrics()
        save_metrics_snapshot(rate, "pre", pre_metrics)
        summary_path = run_k6(rate)
        wait_for_drain()
        post_metrics = fetch_metrics()
        save_metrics_snapshot(rate, "post", post_metrics)

        summary = read_summary(summary_path)
        queue_p95 = histogram_delta_p95(pre_metrics, post_metrics, "faceid_async_job_queue_delay_ms")
        processing_p95 = histogram_delta_p95(pre_metrics, post_metrics, "faceid_async_job_processing_ms")
        total_latency_p95 = histogram_delta_p95(pre_metrics, post_metrics, "faceid_async_job_total_latency_ms")
        queue_pending_post = parse_gauge(post_metrics, "faceid_queue_jobs_pending")
        inflight_post = parse_gauge(post_metrics, "faceid_verify_inflight_current")
        http_inflight_post = parse_gauge(post_metrics, "faceid_verify_async_http_inflight")
        run_valid_for_completion_curve = (
            summary.completed_eventually_pct > 0.0
            and (queue_pending_post or 0.0) == 0.0
            and (inflight_post or 0.0) == 0.0
        )

        rows.append(
            {
                "rate": rate,
                "completion_failed_pct": summary.completion_failed_pct,
                "completed_eventually_pct": summary.completed_eventually_pct,
                "client_iteration_p95_ms": summary.client_iteration_p95_ms,
                "client_terminal_e2e_p95_ms": summary.client_terminal_e2e_p95_ms,
                "queue_p95_ms": queue_p95,
                "processing_p95_ms": processing_p95,
                "total_latency_p95_ms": total_latency_p95,
                "queue_pending_post": queue_pending_post,
                "inflight_post": inflight_post,
                "http_inflight_post": http_inflight_post,
                "run_valid_for_completion_curve": run_valid_for_completion_curve,
            }
        )
        print(
            json.dumps(
                {
                    "rate": rate,
                    "completion_failed_pct": round(summary.completion_failed_pct, 3),
                    "completed_eventually_pct": round(summary.completed_eventually_pct, 3),
                    "client_iteration_p95_ms": None
                    if summary.client_iteration_p95_ms is None
                    else round(summary.client_iteration_p95_ms, 3),
                    "client_terminal_e2e_p95_ms": None
                    if summary.client_terminal_e2e_p95_ms is None
                    else round(summary.client_terminal_e2e_p95_ms, 3),
                    "queue_p95_ms": None if queue_p95 is None else round(queue_p95, 3),
                    "processing_p95_ms": None if processing_p95 is None else round(processing_p95, 3),
                    "total_latency_p95_ms": None if total_latency_p95 is None else round(total_latency_p95, 3),
                    "queue_pending_post": None if queue_pending_post is None else round(queue_pending_post, 3),
                    "inflight_post": None if inflight_post is None else round(inflight_post, 3),
                    "http_inflight_post": None if http_inflight_post is None else round(http_inflight_post, 3),
                    "run_valid_for_completion_curve": run_valid_for_completion_curve,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    print("\nFinal Table", flush=True)
    print("| rate | completion_failed % | completed_eventually % | client iteration p95 ms | client terminal e2e p95 ms | queue p95 ms | processing p95 ms | total_latency p95 ms | queue_pending_post | inflight_post | http_inflight_post | valid_for_completion_curve |", flush=True)
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|", flush=True)
    for row in rows:
        print(
            "| {rate} | {completion_failed_pct:.2f} | {completed_eventually_pct:.2f} | {client_iteration_p95_ms:.2f} | {client_terminal_e2e_p95_ms:.2f} | {queue_p95_ms:.2f} | {processing_p95_ms:.2f} | {total_latency_p95_ms:.2f} | {queue_pending_post:.2f} | {inflight_post:.2f} | {http_inflight_post:.2f} | {run_valid_for_completion_curve} |".format(
                rate=row["rate"],
                completion_failed_pct=row["completion_failed_pct"],
                completed_eventually_pct=row["completed_eventually_pct"],
                client_iteration_p95_ms=row["client_iteration_p95_ms"] or 0.0,
                client_terminal_e2e_p95_ms=row["client_terminal_e2e_p95_ms"] or 0.0,
                queue_p95_ms=row["queue_p95_ms"] or 0.0,
                processing_p95_ms=row["processing_p95_ms"] or 0.0,
                total_latency_p95_ms=row["total_latency_p95_ms"] or 0.0,
                queue_pending_post=row["queue_pending_post"] or 0.0,
                inflight_post=row["inflight_post"] or 0.0,
                http_inflight_post=row["http_inflight_post"] or 0.0,
                run_valid_for_completion_curve=row["run_valid_for_completion_curve"],
            ),
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
