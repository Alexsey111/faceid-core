import math
import os
import logging
import subprocess
import time

import requests

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9190")
COMPOSE_SERVICE = "worker"

MIN_WORKERS = 2
MAX_WORKERS = 6

SCALE_OUT_COOLDOWN = 120
SCALE_IN_COOLDOWN = 600

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autoscaler")

last_scale_time = 0


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


def get_queue_delay_p95():
    return query_prometheus(
        "histogram_quantile(0.95, sum by (le) (rate(faceid_queue_delay_ms_bucket[5m])))"
    )


def get_queue_length():
    # Queue length is emitted by all worker replicas, so aggregate to one scalar.
    return query_prometheus("max(faceid_queue_jobs_pending)")


def get_cpu_usage():
    # Average CPU across all verify_worker containers.
    return query_prometheus(
        'avg(rate(container_cpu_usage_seconds_total{container=~"verify_worker.*"}[5m])) * 100'
    )


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
                "up",
                "-d",
                "--scale",
                f"{COMPOSE_SERVICE}={n}",
                COMPOSE_SERVICE,
            ],
            check=True,
        )
    except Exception as e:
        logger.error(f"scale_failed: {e}")


def decide_scale(current, queue_delay, queue_len, cpu):
    # --- SCALE OUT ---
    if queue_delay > 1000 or queue_len > 5 or cpu > 75:
        return min(current + 1, MAX_WORKERS)

    # --- SCALE IN ---
    if queue_delay < 200 and queue_len < max(1, current - 1) and cpu < 40:
        return max(current - 1, MIN_WORKERS)

    return current


def main_loop():
    global last_scale_time

    while True:
        try:
            current = get_current_workers()

            queue_delay = get_queue_delay_p95()
            queue_len = get_queue_length()
            cpu = get_cpu_usage()

            logger.info(
                f"workers={current} "
                f"queue_delay={queue_delay:.1f}ms "
                f"queue_len={queue_len:.1f} "
                f"cpu={cpu:.1f}%"
            )

            desired = decide_scale(current, queue_delay, queue_len, cpu)

            now = time.time()

            if desired > current:
                if now - last_scale_time > SCALE_OUT_COOLDOWN:
                    scale_workers(desired)
                    last_scale_time = now

            elif desired < current:
                if now - last_scale_time > SCALE_IN_COOLDOWN:
                    scale_workers(desired)
                    last_scale_time = now

        except Exception as e:
            logger.exception(f"autoscaler_loop_error: {e}")

        time.sleep(30)


if __name__ == "__main__":
    main_loop()
