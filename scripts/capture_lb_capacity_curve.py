from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import redis


ROOT = Path(__file__).resolve().parents[1]
K6_SCRIPT = ROOT / "load_test_async.js"
DATA_FILE = ROOT / "tests" / "data" / "person1_small.b64.txt"
SUMMARY_DIR = ROOT / "benchmarks" / "lb_capacity_curve"
COMPOSE_FILE = Path(
    os.environ.get(
        "COMPOSE_FILE",
        str(ROOT / "infrastructure" / "docker-compose.yml"),
    )
)
EXPECTED_WORKERS = int(os.environ.get("EXPECTED_WORKERS", "8"))
JOB_ROW_PREFIX = "JOB_ROW "
K6_LOG_PATH = SUMMARY_DIR / "k6.log"
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8080")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.environ.get("QUEUE_NAME", "face_verify_queue")
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
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


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


def run_compose(*args: str) -> str:
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), *args]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return (proc.stdout or "") + (proc.stderr or "")


def save_text_snapshot(name: str, text: str) -> Path:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out = SUMMARY_DIR / name
    out.write_text(text, encoding="utf-8", errors="replace")
    return out


def count_scaled_workers() -> int:
    text = run_compose("ps", "worker")
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return 0
    return len(lines) - 1


def save_metrics_snapshot(rate: int, phase: str, metrics_text: str) -> Path:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = SUMMARY_DIR / f"r{rate}_{phase}.metrics"
    snapshot_path.write_text(metrics_text, encoding="utf-8")
    return snapshot_path


def extract_job_rows_from_k6_log(k6_log_path: Path, out_path: Path) -> int:
    count = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with k6_log_path.open("r", encoding="utf-8", errors="replace") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            marker = JOB_ROW_PREFIX
            idx = line.find(marker)
            if idx == -1:
                continue
            payload = line[idx + len(marker):].strip()
            if not payload:
                continue
            if " source=" in payload:
                payload = payload.split(" source=", 1)[0].rstrip()
            if payload.endswith('"'):
                payload = payload[:-1]
            if payload.startswith('"'):
                payload = payload[1:]
            payload = payload.replace(r"\"", '"')
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    return count


def save_job_rows(rate: int, k6_log_path: Path) -> tuple[Path, int]:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = SUMMARY_DIR / f"r{rate}_job_rows.jsonl"
    count = extract_job_rows_from_k6_log(k6_log_path, rows_path)
    return rows_path, count


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


def parse_metric_value(metrics_text: str, metric_name: str) -> float | None:
    re_ = re.compile(rf"^{re.escape(metric_name)}(?:\{{[^}}]*\}})? ([0-9.]+)$", re.MULTILINE)
    matches = re_.findall(metrics_text)
    if not matches:
        return None
    return max(float(value) for value in matches)


def counter_delta(pre_text: str, post_text: str, metric_name: str) -> float | None:
    pre_value = parse_metric_value(pre_text, metric_name)
    post_value = parse_metric_value(post_text, metric_name)
    if pre_value is None or post_value is None:
        return None
    return max(0.0, post_value - pre_value)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    lower_weight = upper - index
    upper_weight = index - lower
    return ordered[lower] * lower_weight + ordered[upper] * upper_weight


def sample_active_batches(stop_event: threading.Event, sink: list[float], interval_s: float = 1.0) -> None:
    while not stop_event.is_set():
        try:
            metrics = fetch_metrics()
            value = parse_gauge(metrics, "faceid_worker_active_batches")
            if value is not None:
                sink.append(value)
        except Exception:
            pass
        stop_event.wait(interval_s)


def fetch_runtime_state() -> tuple[float, float]:
    queue_pending = float(cast(int, redis_client.llen(QUEUE_NAME) or 0))
    inflight = float(cast(int | str, redis_client.get("inflight_jobs") or 0))
    return queue_pending, inflight


def histogram_delta_p95(pre_text: str, post_text: str, metric_name: str) -> float | None:
    return histogram_delta_quantile(pre_text, post_text, metric_name, 0.95)


def histogram_delta_quantile(
    pre_text: str,
    post_text: str,
    metric_name: str,
    quantile: float,
) -> float | None:
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

    target = total * quantile
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
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    K6_LOG_PATH.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8", errors="replace")
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr, flush=True)
    _, row_count = save_job_rows(rate, K6_LOG_PATH)
    print(f"job_rows={row_count}", flush=True)
    if proc.returncode != 0:
        print(f"k6 exit_code={proc.returncode}", file=sys.stderr, flush=True)
    return summary_path


def wait_for_drain() -> None:
    deadline = time.time() + DRAIN_TIMEOUT_S
    stable = 0
    while time.time() < deadline:
        metrics = fetch_metrics()
        queue_pending, inflight = fetch_runtime_state()
        http_inflight = parse_gauge(metrics, "faceid_verify_async_http_inflight") or 0.0

        if queue_pending <= 0 and inflight <= 0 and http_inflight <= 0.0:
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
    save_text_snapshot("compose_config.txt", run_compose("config"))
    save_text_snapshot("compose_ps_initial.txt", run_compose("ps"))
    worker_count = count_scaled_workers()
    print(f"detected_worker_containers={worker_count}", flush=True)
    if EXPECTED_WORKERS > 0 and worker_count != EXPECTED_WORKERS:
        raise SystemExit(
            f"Expected {EXPECTED_WORKERS} worker containers, got {worker_count}. "
            f"Start benchmark with explicit scale first."
        )

    rows = []
    for rate in RATES:
        pre_metrics = fetch_metrics()
        save_metrics_snapshot(rate, "pre", pre_metrics)
        save_text_snapshot(f"r{rate}_compose_ps_pre.txt", run_compose("ps"))
        active_batches_samples: list[float] = []
        stop_event = threading.Event()
        sampler = threading.Thread(
            target=sample_active_batches,
            args=(stop_event, active_batches_samples),
            daemon=True,
        )
        sampler.start()
        summary_path = run_k6(rate)
        stop_event.set()
        sampler.join(timeout=5)
        wait_for_drain()
        post_metrics = fetch_metrics()
        save_metrics_snapshot(rate, "post", post_metrics)
        save_text_snapshot(f"r{rate}_compose_ps_post.txt", run_compose("ps"))

        summary = read_summary(summary_path)
        queue_pop_p50 = histogram_delta_quantile(pre_metrics, post_metrics, "faceid_queue_pop_latency_ms", 0.50)
        queue_pop_p95 = histogram_delta_quantile(pre_metrics, post_metrics, "faceid_queue_pop_latency_ms", 0.95)
        enqueue_to_worker_attempt_p50 = histogram_delta_quantile(
            pre_metrics, post_metrics, "faceid_queue_enqueue_to_worker_attempt_ms", 0.50
        )
        enqueue_to_worker_attempt_p95 = histogram_delta_quantile(
            pre_metrics, post_metrics, "faceid_queue_enqueue_to_worker_attempt_ms", 0.95
        )
        worker_attempt_to_claim_success_p50 = histogram_delta_quantile(
            pre_metrics, post_metrics, "faceid_queue_worker_attempt_to_claim_success_ms", 0.50
        )
        worker_attempt_to_claim_success_p95 = histogram_delta_quantile(
            pre_metrics, post_metrics, "faceid_queue_worker_attempt_to_claim_success_ms", 0.95
        )
        batch_size_p50 = histogram_delta_quantile(pre_metrics, post_metrics, "faceid_queue_batch_size", 0.50)
        batch_size_p95 = histogram_delta_quantile(pre_metrics, post_metrics, "faceid_queue_batch_size", 0.95)
        jobs_per_pop_p50 = histogram_delta_quantile(pre_metrics, post_metrics, "faceid_queue_jobs_per_pop", 0.50)
        jobs_per_pop_p95 = histogram_delta_quantile(pre_metrics, post_metrics, "faceid_queue_jobs_per_pop", 0.95)
        idle_gap_p50 = histogram_delta_quantile(pre_metrics, post_metrics, "faceid_worker_idle_gap_ms", 0.50)
        idle_gap_p95 = histogram_delta_quantile(pre_metrics, post_metrics, "faceid_worker_idle_gap_ms", 0.95)
        worker_semaphore_wait_p50 = histogram_delta_quantile(
            pre_metrics, post_metrics, "faceid_worker_semaphore_wait_ms", 0.50
        )
        worker_semaphore_wait_p95 = histogram_delta_quantile(
            pre_metrics, post_metrics, "faceid_worker_semaphore_wait_ms", 0.95
        )
        assignment_delay_p50 = histogram_delta_quantile(pre_metrics, post_metrics, "faceid_queue_assignment_delay_ms", 0.50)
        assignment_delay_p95 = histogram_delta_quantile(pre_metrics, post_metrics, "faceid_queue_assignment_delay_ms", 0.95)
        first_claim_p50 = histogram_delta_quantile(pre_metrics, post_metrics, "faceid_queue_time_to_first_claim_ms", 0.50)
        first_claim_p95 = histogram_delta_quantile(pre_metrics, post_metrics, "faceid_queue_time_to_first_claim_ms", 0.95)
        claim_to_fill_p50 = histogram_delta_quantile(pre_metrics, post_metrics, "faceid_queue_claim_to_batch_fill_ms", 0.50)
        claim_to_fill_p95 = histogram_delta_quantile(pre_metrics, post_metrics, "faceid_queue_claim_to_batch_fill_ms", 0.95)
        fill_to_start_p50 = histogram_delta_quantile(pre_metrics, post_metrics, "faceid_queue_batch_ready_to_processing_start_ms", 0.50)
        fill_to_start_p95 = histogram_delta_quantile(pre_metrics, post_metrics, "faceid_queue_batch_ready_to_processing_start_ms", 0.95)
        claim_attempts_delta = counter_delta(pre_metrics, post_metrics, "faceid_queue_claim_attempts_total")
        claim_success_delta = counter_delta(pre_metrics, post_metrics, "faceid_queue_claim_success_total")
        queue_p95 = histogram_delta_p95(pre_metrics, post_metrics, "faceid_async_job_queue_delay_ms")
        processing_p95 = histogram_delta_p95(pre_metrics, post_metrics, "faceid_async_job_processing_ms")
        total_latency_p95 = histogram_delta_p95(pre_metrics, post_metrics, "faceid_async_job_total_latency_ms")
        queue_pending_post = parse_gauge(post_metrics, "faceid_queue_jobs_pending")
        inflight_post = parse_gauge(post_metrics, "faceid_verify_inflight_current")
        http_inflight_post = parse_gauge(post_metrics, "faceid_verify_async_http_inflight")
        queue_pending_post, inflight_post = fetch_runtime_state()
        worker_active_batches_post = parse_gauge(post_metrics, "faceid_worker_active_batches")
        worker_active_batches_p50 = percentile(active_batches_samples, 0.50)
        worker_active_batches_p95 = percentile(active_batches_samples, 0.95)
        worker_active_batches_max = max(active_batches_samples) if active_batches_samples else None
        run_valid_for_completion_curve = (
            summary.completed_eventually_pct > 0.0
            and queue_pending_post == 0.0
            and inflight_post == 0.0
        )

        rows.append(
            {
                "rate": rate,
                "completion_failed_pct": summary.completion_failed_pct,
                "completed_eventually_pct": summary.completed_eventually_pct,
                "queue_pop_p50_ms": queue_pop_p50,
                "queue_pop_p95_ms": queue_pop_p95,
                "enqueue_to_worker_attempt_p50_ms": enqueue_to_worker_attempt_p50,
                "enqueue_to_worker_attempt_p95_ms": enqueue_to_worker_attempt_p95,
                "worker_attempt_to_claim_success_p50_ms": worker_attempt_to_claim_success_p50,
                "worker_attempt_to_claim_success_p95_ms": worker_attempt_to_claim_success_p95,
                "batch_size_p50": batch_size_p50,
                "batch_size_p95": batch_size_p95,
                "jobs_per_pop_p50": jobs_per_pop_p50,
                "jobs_per_pop_p95": jobs_per_pop_p95,
                "idle_gap_p50_ms": idle_gap_p50,
                "idle_gap_p95_ms": idle_gap_p95,
                "worker_semaphore_wait_p50_ms": worker_semaphore_wait_p50,
                "worker_semaphore_wait_p95_ms": worker_semaphore_wait_p95,
                "assignment_delay_p50_ms": assignment_delay_p50,
                "assignment_delay_p95_ms": assignment_delay_p95,
                "first_claim_p50_ms": first_claim_p50,
                "first_claim_p95_ms": first_claim_p95,
                "claim_to_fill_p50_ms": claim_to_fill_p50,
                "claim_to_fill_p95_ms": claim_to_fill_p95,
                "fill_to_start_p50_ms": fill_to_start_p50,
                "fill_to_start_p95_ms": fill_to_start_p95,
                "claim_attempts_delta": claim_attempts_delta,
                "claim_success_delta": claim_success_delta,
                "worker_active_batches_post": worker_active_batches_post,
                "worker_active_batches_p50": worker_active_batches_p50,
                "worker_active_batches_p95": worker_active_batches_p95,
                "worker_active_batches_max": worker_active_batches_max,
                "client_iteration_p95_ms": summary.client_iteration_p95_ms,
                "client_terminal_e2e_p95_ms": summary.client_terminal_e2e_p95_ms,
                "queue_p95_ms": queue_p95,
                "processing_p95_ms": processing_p95,
                "total_latency_p95_ms": total_latency_p95,
                "queue_pending_post": queue_pending_post,
                "inflight_post": inflight_post,
                "http_inflight_post": http_inflight_post,
                "run_valid_for_completion_curve": run_valid_for_completion_curve,
                "compose_worker_count": worker_count,
            }
        )
        print(
            json.dumps(
                {
                    "rate": rate,
                    "completion_failed_pct": round(summary.completion_failed_pct, 3),
                    "completed_eventually_pct": round(summary.completed_eventually_pct, 3),
                    "queue_pop_p50_ms": None if queue_pop_p50 is None else round(queue_pop_p50, 3),
                    "queue_pop_p95_ms": None if queue_pop_p95 is None else round(queue_pop_p95, 3),
                    "enqueue_to_worker_attempt_p50_ms": None
                    if enqueue_to_worker_attempt_p50 is None
                    else round(enqueue_to_worker_attempt_p50, 3),
                    "enqueue_to_worker_attempt_p95_ms": None
                    if enqueue_to_worker_attempt_p95 is None
                    else round(enqueue_to_worker_attempt_p95, 3),
                    "worker_attempt_to_claim_success_p50_ms": None
                    if worker_attempt_to_claim_success_p50 is None
                    else round(worker_attempt_to_claim_success_p50, 3),
                    "worker_attempt_to_claim_success_p95_ms": None
                    if worker_attempt_to_claim_success_p95 is None
                    else round(worker_attempt_to_claim_success_p95, 3),
                    "batch_size_p50": None if batch_size_p50 is None else round(batch_size_p50, 3),
                    "batch_size_p95": None if batch_size_p95 is None else round(batch_size_p95, 3),
                    "jobs_per_pop_p50": None if jobs_per_pop_p50 is None else round(jobs_per_pop_p50, 3),
                    "jobs_per_pop_p95": None if jobs_per_pop_p95 is None else round(jobs_per_pop_p95, 3),
                    "idle_gap_p50_ms": None if idle_gap_p50 is None else round(idle_gap_p50, 3),
                    "idle_gap_p95_ms": None if idle_gap_p95 is None else round(idle_gap_p95, 3),
                    "worker_semaphore_wait_p50_ms": None
                    if worker_semaphore_wait_p50 is None
                    else round(worker_semaphore_wait_p50, 3),
                    "worker_semaphore_wait_p95_ms": None
                    if worker_semaphore_wait_p95 is None
                    else round(worker_semaphore_wait_p95, 3),
                    "assignment_delay_p50_ms": None
                    if assignment_delay_p50 is None
                    else round(assignment_delay_p50, 3),
                    "assignment_delay_p95_ms": None
                    if assignment_delay_p95 is None
                    else round(assignment_delay_p95, 3),
                    "first_claim_p50_ms": None if first_claim_p50 is None else round(first_claim_p50, 3),
                    "first_claim_p95_ms": None if first_claim_p95 is None else round(first_claim_p95, 3),
                    "claim_to_fill_p50_ms": None if claim_to_fill_p50 is None else round(claim_to_fill_p50, 3),
                    "claim_to_fill_p95_ms": None if claim_to_fill_p95 is None else round(claim_to_fill_p95, 3),
                    "fill_to_start_p50_ms": None if fill_to_start_p50 is None else round(fill_to_start_p50, 3),
                    "fill_to_start_p95_ms": None if fill_to_start_p95 is None else round(fill_to_start_p95, 3),
                    "claim_attempts_delta": None if claim_attempts_delta is None else round(claim_attempts_delta, 3),
                    "claim_success_delta": None if claim_success_delta is None else round(claim_success_delta, 3),
                    "worker_active_batches_post": None
                    if worker_active_batches_post is None
                    else round(worker_active_batches_post, 3),
                    "worker_active_batches_p50": None
                    if worker_active_batches_p50 is None
                    else round(worker_active_batches_p50, 3),
                    "worker_active_batches_p95": None
                    if worker_active_batches_p95 is None
                    else round(worker_active_batches_p95, 3),
                    "worker_active_batches_max": None
                    if worker_active_batches_max is None
                    else round(worker_active_batches_max, 3),
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
    print("| rate | completion_failed % | completed_eventually % | queue_pop p50/p95 ms | enqueue->attempt p50/p95 ms | attempt->success p50/p95 ms | batch_size p50/p95 | jobs_per_pop p50/p95 | idle_gap p50/p95 ms | sem_wait p50/p95 ms | assignment_delay p50/p95 ms | first_claim p50/p95 ms | claim_to_fill p50/p95 ms | fill_to_start p50/p95 ms | claim_attempts | claim_success | active_batches p50/p95/max | client iteration p95 ms | client terminal e2e p95 ms | queue p95 ms | processing p95 ms | total_latency p95 ms | queue_pending_post | inflight_post | http_inflight_post | valid_for_completion_curve |", flush=True)
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|", flush=True)
    for row in rows:
        print(
            "| {rate} | {completion_failed_pct:.2f} | {completed_eventually_pct:.2f} | {queue_pop_p50_ms:.2f}/{queue_pop_p95_ms:.2f} | {enqueue_to_worker_attempt_p50_ms:.2f}/{enqueue_to_worker_attempt_p95_ms:.2f} | {worker_attempt_to_claim_success_p50_ms:.2f}/{worker_attempt_to_claim_success_p95_ms:.2f} | {batch_size_p50:.2f}/{batch_size_p95:.2f} | {jobs_per_pop_p50:.2f}/{jobs_per_pop_p95:.2f} | {idle_gap_p50_ms:.2f}/{idle_gap_p95_ms:.2f} | {worker_semaphore_wait_p50_ms:.2f}/{worker_semaphore_wait_p95_ms:.2f} | {assignment_delay_p50_ms:.2f}/{assignment_delay_p95_ms:.2f} | {first_claim_p50_ms:.2f}/{first_claim_p95_ms:.2f} | {claim_to_fill_p50_ms:.2f}/{claim_to_fill_p95_ms:.2f} | {fill_to_start_p50_ms:.2f}/{fill_to_start_p95_ms:.2f} | {claim_attempts_delta:.2f} | {claim_success_delta:.2f} | {worker_active_batches_p50:.2f}/{worker_active_batches_p95:.2f}/{worker_active_batches_max:.2f} | {client_iteration_p95_ms:.2f} | {client_terminal_e2e_p95_ms:.2f} | {queue_p95_ms:.2f} | {processing_p95_ms:.2f} | {total_latency_p95_ms:.2f} | {queue_pending_post:.2f} | {inflight_post:.2f} | {http_inflight_post:.2f} | {run_valid_for_completion_curve} |".format(
                rate=row["rate"],
                completion_failed_pct=row["completion_failed_pct"],
                completed_eventually_pct=row["completed_eventually_pct"],
                queue_pop_p50_ms=row["queue_pop_p50_ms"] or 0.0,
                queue_pop_p95_ms=row["queue_pop_p95_ms"] or 0.0,
                enqueue_to_worker_attempt_p50_ms=row["enqueue_to_worker_attempt_p50_ms"] or 0.0,
                enqueue_to_worker_attempt_p95_ms=row["enqueue_to_worker_attempt_p95_ms"] or 0.0,
                worker_attempt_to_claim_success_p50_ms=row["worker_attempt_to_claim_success_p50_ms"] or 0.0,
                worker_attempt_to_claim_success_p95_ms=row["worker_attempt_to_claim_success_p95_ms"] or 0.0,
                batch_size_p50=row["batch_size_p50"] or 0.0,
                batch_size_p95=row["batch_size_p95"] or 0.0,
                jobs_per_pop_p50=row["jobs_per_pop_p50"] or 0.0,
                jobs_per_pop_p95=row["jobs_per_pop_p95"] or 0.0,
                idle_gap_p50_ms=row["idle_gap_p50_ms"] or 0.0,
                idle_gap_p95_ms=row["idle_gap_p95_ms"] or 0.0,
                worker_semaphore_wait_p50_ms=row["worker_semaphore_wait_p50_ms"] or 0.0,
                worker_semaphore_wait_p95_ms=row["worker_semaphore_wait_p95_ms"] or 0.0,
                assignment_delay_p50_ms=row["assignment_delay_p50_ms"] or 0.0,
                assignment_delay_p95_ms=row["assignment_delay_p95_ms"] or 0.0,
                first_claim_p50_ms=row["first_claim_p50_ms"] or 0.0,
                first_claim_p95_ms=row["first_claim_p95_ms"] or 0.0,
                claim_to_fill_p50_ms=row["claim_to_fill_p50_ms"] or 0.0,
                claim_to_fill_p95_ms=row["claim_to_fill_p95_ms"] or 0.0,
                fill_to_start_p50_ms=row["fill_to_start_p50_ms"] or 0.0,
                fill_to_start_p95_ms=row["fill_to_start_p95_ms"] or 0.0,
                claim_attempts_delta=row["claim_attempts_delta"] or 0.0,
                claim_success_delta=row["claim_success_delta"] or 0.0,
                worker_active_batches_p50=row["worker_active_batches_p50"] or 0.0,
                worker_active_batches_p95=row["worker_active_batches_p95"] or 0.0,
                worker_active_batches_max=row["worker_active_batches_max"] or 0.0,
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

    save_text_snapshot("compose_config_post.txt", run_compose("config"))
    save_text_snapshot("compose_ps_post.txt", run_compose("ps"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
