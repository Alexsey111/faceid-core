from time import perf_counter

from fastapi import Request

from app.core.middleware import resolve_endpoint
from app.monitoring.metrics import (
    ERROR_COUNTER,
    INPROGRESS_REQUESTS,
    REQUEST_COUNTER,
    REQUEST_LATENCY,
    VERIFY_ASYNC_ADMISSION_MS,
    VERIFY_ASYNC_HTTP_INFLIGHT,
    VERIFY_ASYNC_MIDDLEWARE_HITS_TOTAL,
    VERIFY_ASYNC_REQUEST_SIZE_BYTES,
    VERIFY_ASYNC_STATUS_TOTAL,
)


def _is_verify_async_request(request: Request) -> bool:
    method = request.method
    path = (request.scope.get("path") or request.url.path or "").rstrip("/")
    normalized = path.rstrip("/")
    return method == "POST" and normalized.endswith("/verify_async")


async def metrics_middleware(request: Request, call_next):
    raw_path = request.url.path
    method = request.method

    if raw_path == "/metrics":
        return await call_next(request)

    is_verify_async = _is_verify_async_request(request)

    started = perf_counter()
    INPROGRESS_REQUESTS.inc()

    if is_verify_async:
        VERIFY_ASYNC_MIDDLEWARE_HITS_TOTAL.inc()
        VERIFY_ASYNC_HTTP_INFLIGHT.inc()

        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit():
            VERIFY_ASYNC_REQUEST_SIZE_BYTES.observe(int(content_length))

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
        endpoint = resolve_endpoint(request)

        REQUEST_COUNTER.labels(
            endpoint=endpoint,
            method=method,
            status=str(status_code),
        ).inc()

        REQUEST_LATENCY.labels(
            endpoint=endpoint,
        ).observe(perf_counter() - started)

        if is_verify_async:
            VERIFY_ASYNC_STATUS_TOTAL.labels(status=str(status_code)).inc()
            VERIFY_ASYNC_ADMISSION_MS.observe((perf_counter() - started) * 1000)
            VERIFY_ASYNC_HTTP_INFLIGHT.dec()

        INPROGRESS_REQUESTS.dec()
