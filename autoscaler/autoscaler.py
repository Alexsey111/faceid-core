# autoscaler\autoscaler.py

import math
import os
import logging
import subprocess
import time
from typing import cast

import requests
import redis

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9190")
COMPOSE_SERVICE = "worker"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

REDIS_POOL = redis.ConnectionPool.from_url(
    REDIS_URL,
    decode_responses=True,
    max_connections=50,
)

redis_client = redis.Redis(connection_pool=REDIS_POOL)

MIN_WORKERS = 2
MAX_WORKERS = 8
MIN_STABLE_WORKERS = 4

SCALE_OUT_COOLDOWN = 10
SCALE_IN_COOLDOWN = 90
REQUIRED_LOW_DELAY_CYCLES = 3
MIN_UPTIME_BEFORE_SCALE_IN = 60  # seconds
SCALE_IN_GRACE_PERIOD = 120  # seconds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autoscaler")

last_scale_time = 0
low_delay_counter = 0


def query_prometheus(expr: str) -> float:
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": expr},
            timeout=2,
        )
        data = resp.json()
        if data.get("status") != "success":
            raise RuntimeError(f"prometheus returned {data.get('status')}")

        result = data["data"]["result"]
        if not result:
            return 0.0
        value = float(result[0]["value"][1])
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value
    except Exception as e:
        logger.warning(f"prometheus_query_failed: {e}")
        return 0.0


def get_queue_delay_ms():
    try:
        raw = redis_client.get("metrics:queue_delay_ms")
        if raw is not None:
            value = float(cast(str, raw))
            if not math.isnan(value) and not math.isinf(value):
                return value
    except Exception as e:
        logger.warning(f"redis_queue_delay_failed: {e}")

    return query_prometheus(
        "histogram_quantile(0.95, sum by (le) (rate(faceid_async_job_queue_delay_ms_bucket[5m])))"
    )


def get_queue_length():
    # Queue length is emitted by all worker replicas, so aggregate to one scalar.
    return query_prometheus("max(faceid_queue_jobs_pending)")


def get_current_workers():
    try:
        result = subprocess.check_output(
            [
                "docker",
                "ps",
                "--filter",
                "label=com.docker.compose.project=faceid-core",
                "--filter",
                "label=com.docker.compose.service=worker",
                "--format",
                "{{.Names}}",
            ]
        )
        workers = result.decode().strip().split("\n")
        return len([w for w in workers if w])
    except Exception as e:
        logger.error(f"failed_to_get_workers: {e}")
        return MIN_WORKERS


def scale_workers(n: int):
    logger.warning(f"SCALING -> {n} workers")

    try:
        subprocess.run(
            [
                "docker",
                "compose",
                "-p",
                "faceid-core",
                "-f",
                "/workspace/docker-compose.yml",
                "scale",
                f"{COMPOSE_SERVICE}={n}",
            ],
            check=True,
        )
    except Exception as e:
        logger.error(f"scale_failed: {e}")


def decide_scale(current, queue_delay, queue_len):
    global low_delay_counter

    # --- SCALE OUT ---
    if queue_delay > 200 or queue_len > 20:
        low_delay_counter = 0

        if queue_delay > 1000 or queue_len > 50:
            return min(current + 2, MAX_WORKERS)

        return min(current + 1, MAX_WORKERS)

    # --- TRACK LOW DELAY ---
    if queue_delay < 20 and queue_len < 2:
        low_delay_counter += 1
    else:
        low_delay_counter = 0

    # --- SCALE IN (only if stable low) ---
    if current <= MIN_STABLE_WORKERS:
        return current

    if low_delay_counter >= 5:
        return max(current - 1, MIN_WORKERS)

    return current


def main_loop():
    global last_scale_time

    while True:
        try:
            current = get_current_workers()

            queue_delay = get_queue_delay_ms()
            queue_len = get_queue_length()

            logger.info(
                f"workers={current} "
                f"queue_delay={queue_delay:.1f}ms "
                f"queue_len={queue_len:.1f}"
            )

            desired = decide_scale(current, queue_delay, queue_len)

            now = time.time()

            if desired > current:
                if now - last_scale_time > SCALE_OUT_COOLDOWN:
                    scale_workers(desired)
                    last_scale_time = now

            elif desired < current:
                if now - last_scale_time < SCALE_IN_GRACE_PERIOD:
                    continue
                if now - last_scale_time > SCALE_IN_COOLDOWN:
                    if now - last_scale_time > MIN_UPTIME_BEFORE_SCALE_IN:
                        scale_workers(desired)
                        last_scale_time = now

        except Exception as e:
            logger.exception(f"autoscaler_loop_error: {e}")

        time.sleep(5)


if __name__ == "__main__":
    main_loop()
