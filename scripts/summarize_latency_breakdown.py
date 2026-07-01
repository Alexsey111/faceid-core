from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STAGE_ORDER = [
    "admission",
    "enqueue",
    "queue_wait",
    "batch_wait",
    "worker_semaphore_wait",
    "pipeline_total",
    "preprocess",
    "detect",
    "align",
    "encode",
    "search",
    "liveness",
    "decision",
    "result_write",
    "job_total_server",
    "server_wait",
    "wait_lookup",
    "result_visible_age",
    "wait_empty_cycles",
    "client_terminal_e2e",
    "poll_overhead",
]

STAGE_KEYS = {
    "admission": {"admission_ms"},
    "enqueue": {"enqueue_ms"},
    "queue_wait": {"queue_wait_ms"},
    "batch_wait": {"batch_wait_ms"},
    "worker_semaphore_wait": {"worker_semaphore_wait_ms"},
    "pipeline_total": {"pipeline_total_ms"},
    "preprocess": {"preprocess_ms"},
    "detect": {"detect_ms"},
    "align": {"align_ms"},
    "encode": {"encode_ms"},
    "search": {"search_ms"},
    "liveness": {"liveness_ms"},
    "decision": {"decision_ms"},
    "result_write": {"result_write_ms"},
    "job_total_server": {"job_total_server_ms"},
    "server_wait": {"server_wait_ms", "wait_block_ms", "wait_route_ms"},
    "wait_lookup": {"wait_lookup_ms", "result_fetch_ms"},
    "result_visible_age": {"result_visible_age_ms"},
    "wait_empty_cycles": {"wait_empty_cycles", "poll_cycles"},
    "client_terminal_e2e": {"client_terminal_e2e_ms", "client_e2e_ms"},
    "poll_overhead": {"poll_overhead_ms"},
}

@dataclass
class StageStats:
    count: int
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    mean_ms: float | None

    def as_row(self, stage: str, total_p95_ms: float | None) -> dict[str, str]:
        share = None
        if total_p95_ms is not None and total_p95_ms > 0 and self.p95_ms is not None:
            share = (self.p95_ms / total_p95_ms) * 100.0
        return {
            "stage": stage,
            "count": str(self.count),
            "p50_ms": fmt(self.p50_ms),
            "p95_ms": fmt(self.p95_ms),
            "p99_ms": fmt(self.p99_ms),
            "mean_ms": fmt(self.mean_ms),
            "share_of_total_p95_pct": fmt(share),
        }


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    if math.isnan(value) or math.isinf(value):
        return ""
    return f"{value:.3f}"


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


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def stage_from_key(key: str) -> str | None:
    normalized = key.split(".")[-1]
    for stage, aliases in STAGE_KEYS.items():
        if normalized in aliases:
            return stage
    return None


def extract_summary_metric_block(block: dict[str, Any], count_hint: int | None = None) -> StageStats | None:
    count = block.get("count")
    if count is None:
        count = block.get("values", {}).get("count") if isinstance(block.get("values"), dict) else None
    if count is None:
        count = count_hint
    count_int = int(count) if isinstance(count, (int, float)) else None
    if count_int is None:
        return None

    def pick(*keys: str) -> float | None:
        for key in keys:
            value = parse_float(block.get(key))
            if value is not None:
                return value
        return None

    return StageStats(
        count=count_int,
        p50_ms=pick("p(50)", "med", "median"),
        p95_ms=pick("p(95)"),
        p99_ms=pick("p(99)"),
        mean_ms=pick("avg", "mean"),
    )


def collect_numeric_samples(obj: Any, stage_samples: dict[str, list[float]]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            stage = stage_from_key(str(key))
            if stage is not None:
                if isinstance(value, dict):
                    collect_numeric_samples(value, stage_samples)
                else:
                    parsed = parse_float(value)
                    if parsed is not None:
                        stage_samples[stage].append(parsed)
            else:
                collect_numeric_samples(value, stage_samples)
    elif isinstance(obj, list):
        for item in obj:
            collect_numeric_samples(item, stage_samples)
    elif isinstance(obj, str):
        stripped = obj.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                collect_numeric_samples(json.loads(stripped), stage_samples)
            except Exception:
                pass


def load_json_file(
    path: Path,
    stage_samples: dict[str, list[float]],
    summary_candidates: list[tuple[Path, StageStats]],
) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return

    if isinstance(data, dict):
        metrics = data.get("metrics")
        if isinstance(metrics, dict):
            summary_block = metrics.get("client_e2e_ms") or metrics.get("client_terminal_e2e_ms")
            if isinstance(summary_block, dict):
                count_hint = None
                for key in ("completed", "result_passes", "terminal_done", "iterations"):
                    block = metrics.get(key)
                    if isinstance(block, dict):
                        hint = parse_float(block.get("count"))
                        if hint is not None:
                            count_hint = int(hint)
                            break
                summary = extract_summary_metric_block(summary_block, count_hint=count_hint)
                if summary is not None:
                    summary_candidates.append((path, summary))
        collect_numeric_samples(data, stage_samples)
    elif isinstance(data, list):
        collect_numeric_samples(data, stage_samples)


def load_jsonl_file(path: Path, stage_samples: dict[str, list[float]]) -> None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                collect_numeric_samples(data, stage_samples)
    except Exception:
        return


def load_csv_file(
    path: Path,
    stage_samples: dict[str, list[float]],
    summary_candidates: list[tuple[Path, StageStats]],
) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                record: dict[str, Any] = {}
                for key, value in row.items():
                    if value is None or value == "":
                        continue
                    stage = stage_from_key(key)
                    parsed = parse_float(value)
                    if stage is not None and parsed is not None:
                        stage_samples[stage].append(parsed)
                        record[stage] = parsed
                        continue
                    if value.lstrip().startswith("{") or value.lstrip().startswith("["):
                        try:
                            parsed_json = json.loads(value)
                            record[key] = parsed_json
                            collect_numeric_samples(parsed_json, stage_samples)
                            continue
                        except Exception:
                            pass
                    record[key] = value
    except Exception:
        return


def summarize_samples(values: list[float]) -> StageStats | None:
    if not values:
        return None
    return StageStats(
        count=len(values),
        p50_ms=percentile(values, 0.50),
        p95_ms=percentile(values, 0.95),
        p99_ms=percentile(values, 0.99),
        mean_ms=statistics.fmean(values),
    )


def derive_poll_overhead(
    client: StageStats | None,
    server: StageStats | None,
    paired_samples: list[float],
) -> StageStats | None:
    if paired_samples:
        return summarize_samples(paired_samples)
    if client is None or server is None:
        return None
    return StageStats(
        count=min(client.count, server.count),
        p50_ms=(
            None
            if client.p50_ms is None or server.p50_ms is None
            else max(0.0, client.p50_ms - server.p50_ms)
        ),
        p95_ms=(
            None
            if client.p95_ms is None or server.p95_ms is None
            else max(0.0, client.p95_ms - server.p95_ms)
        ),
        p99_ms=(
            None
            if client.p99_ms is None or server.p99_ms is None
            else max(0.0, client.p99_ms - server.p99_ms)
        ),
        mean_ms=(
            None
            if client.mean_ms is None or server.mean_ms is None
            else max(0.0, client.mean_ms - server.mean_ms)
        ),
    )


def render_markdown(rows: list[dict[str, str]], notes: list[str]) -> str:
    headers = [
        "stage",
        "count",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "mean_ms",
        "share_of_total_p95_pct",
    ]
    lines = ["# Latency Breakdown", ""]
    for note in notes:
        lines.append(f"- {note}")
    if notes:
        lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(row.get(header, "") or "-" for header in headers)
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize latency breakdown from benchmark artifacts.")
    parser.add_argument("input_dir", type=Path, help="Benchmark run directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for latency_breakdown.csv and latency_breakdown.md (defaults to input_dir)",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_stage_samples: dict[str, list[float]] = defaultdict(list)
    secondary_stage_samples: dict[str, list[float]] = defaultdict(list)
    primary_summary_candidates: list[tuple[Path, StageStats]] = []
    secondary_summary_candidates: list[tuple[Path, StageStats]] = []

    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".jsonl" and path.name.endswith("_job_rows.jsonl"):
            load_jsonl_file(path, primary_stage_samples)
        elif suffix == ".json":
            load_json_file(path, secondary_stage_samples, secondary_summary_candidates)
        elif suffix == ".jsonl":
            load_jsonl_file(path, secondary_stage_samples)
        elif suffix == ".csv":
            load_csv_file(path, secondary_stage_samples, secondary_summary_candidates)

    primary_summary_candidates.sort(
        key=lambda item: (item[1].count, item[0].stat().st_mtime if item[0].exists() else 0.0),
        reverse=True,
    )
    secondary_summary_candidates.sort(
        key=lambda item: (item[1].count, item[0].stat().st_mtime if item[0].exists() else 0.0),
        reverse=True,
    )

    def stage_values(stage: str) -> list[float]:
        values = primary_stage_samples.get(stage, [])
        if values:
            return values
        return secondary_stage_samples.get(stage, [])

    primary_client_summary = summarize_samples(primary_stage_samples.get("client_terminal_e2e", []))
    primary_poll_summary = summarize_samples(primary_stage_samples.get("poll_overhead", []))

    if primary_client_summary is not None:
        client_summary = primary_client_summary
    elif secondary_summary_candidates:
        client_summary = secondary_summary_candidates[0][1]
    else:
        client_summary = summarize_samples(stage_values("client_terminal_e2e"))

    server_summary = summarize_samples(stage_values("job_total_server"))

    if primary_poll_summary is not None:
        poll_summary = primary_poll_summary
    else:
        poll_summary = derive_poll_overhead(client_summary, server_summary, [])

    stage_stats_map: dict[str, StageStats | None] = {}
    for stage in STAGE_ORDER:
        if stage == "client_terminal_e2e":
            stage_stats_map[stage] = client_summary
        elif stage == "poll_overhead":
            stage_stats_map[stage] = poll_summary
        else:
            stage_stats_map[stage] = summarize_samples(stage_values(stage))

    total_p95 = None
    if client_summary is not None and client_summary.p95_ms is not None:
        total_p95 = client_summary.p95_ms
    elif server_summary is not None and server_summary.p95_ms is not None:
        total_p95 = server_summary.p95_ms

    rows: list[dict[str, str]] = []
    for stage in STAGE_ORDER:
        stats = stage_stats_map.get(stage)
        if stats is None:
            rows.append(
                {
                    "stage": stage,
                    "count": "0",
                    "p50_ms": "",
                    "p95_ms": "",
                    "p99_ms": "",
                    "mean_ms": "",
                    "share_of_total_p95_pct": "",
                }
            )
            continue
        rows.append(stats.as_row(stage, total_p95))

    csv_path = output_dir / "latency_breakdown.csv"
    md_path = output_dir / "latency_breakdown.md"

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "stage",
                "count",
                "p50_ms",
                "p95_ms",
                "p99_ms",
                "mean_ms",
                "share_of_total_p95_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    notes = [
        "Primary source for job-level stages is *_job_rows.jsonl.",
        "client_terminal_e2e prefers per-job client_terminal_e2e_ms from *_job_rows.jsonl and falls back to k6 summary only if per-job rows are unavailable.",
        "poll_overhead prefers per-job poll_overhead_ms from *_job_rows.jsonl and falls back to aggregate client minus server latency only if per-job rows are unavailable.",
    ]
    if primary_client_summary is not None:
        notes.insert(0, "Primary client summary source: *_job_rows.jsonl")
    elif secondary_summary_candidates:
        notes.insert(0, f"Primary client summary source: {secondary_summary_candidates[0][0].name}")

    md_path.write_text(render_markdown(rows, notes), encoding="utf-8")
    print(csv_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
