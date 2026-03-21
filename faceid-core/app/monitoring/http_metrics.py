from time import perf_counter
from fastapi import Request

from app.monitoring.metrics import (
    REQUEST_COUNTER,
    REQUEST_LATENCY,
    INPROGRESS_REQUESTS,
    ERROR_COUNTER,
)


async def metrics_middleware(request: Request, call_next):
    path = request.url.path
    method = request.method

    if path == "/metrics":
        return await call_next(request)

    start = perf_counter()
    INPROGRESS_REQUESTS.inc()

    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response

    except Exception as exc:
        ERROR_COUNTER.labels(
            stage="http",
            error_type=type(exc).__name__,
        ).inc()
        raise

    finally:
        REQUEST_COUNTER.labels(
            endpoint=path,
            method=method,
            status=str(status_code),
        ).inc()

        REQUEST_LATENCY.labels(
            endpoint=path,
        ).observe(perf_counter() - start)

        INPROGRESS_REQUESTS.dec()
