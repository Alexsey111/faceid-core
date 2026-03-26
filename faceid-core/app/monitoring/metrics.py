# faceid-core\app\monitoring\metrics.py

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNTER = Counter(
    "faceid_http_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"],
)

FAISS_HIT = Counter(
    "faceid_faiss_hit_total",
    "FAISS hits",
    ["endpoint", "result"],
)

REDIS_HIT = Counter(
    "faceid_redis_hit_total",
    "Redis cache hits",
    ["endpoint", "result"],
)

DB_FALLBACK = Counter(
    "faceid_db_fallback_total",
    "Fallback to DB",
    ["endpoint", "result"],
)

ERROR_COUNTER = Counter(
    "faceid_errors_total",
    "Application errors",
    ["stage", "error_type"],
)

VERIFICATION_RESULT_COUNTER = Counter(
    "faceid_verification_result_total",
    "Verification outcomes",
    ["status", "liveness_passed"],
)

VERIFY_RESULT = Counter(
    "faceid_verify_result_total",
    "Verification results",
    ["result"],
)

VERIFY_RESULT_COUNTER = VERIFY_RESULT

LIVENESS_RESULT_COUNTER = Counter(
    "faceid_liveness_result_total",
    "Liveness outcomes",
    ["result"],
)

LIVENESS_FAIL_COUNT = Counter(
    "faceid_liveness_fail_total",
    "Failed liveness checks",
)

SEARCH_BACKEND_COUNTER = Counter(
    "faceid_search_backend_total",
    "Search backend usage",
    ["backend"],
)

REQUEST_LATENCY = Histogram(
    "faceid_http_request_duration_seconds",
    "API request latency",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0],
)

VERIFY_LATENCY = Histogram(
    "faceid_verify_duration_seconds",
    "Full verify pipeline latency",
)

INPROGRESS_REQUESTS = Gauge(
    "faceid_http_inprogress_requests",
    "HTTP requests currently in progress",
)

PIPELINE_STAGE_DURATION = Histogram(
    "faceid_pipeline_stage_duration_seconds",
    "Verification pipeline stage duration",
    ["stage"],
    buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
)

QUEUE_DELAY_MS = Histogram(
    "faceid_queue_delay_ms",
    "Queue delay before worker starts",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
)

PIPELINE_MS = Histogram(
    "faceid_pipeline_ms",
    "Total pipeline processing time",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
)

DETECT_MS = Histogram(
    "faceid_detect_ms",
    "Face detection time",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
)

ENCODE_MS = Histogram(
    "faceid_encode_ms",
    "Face embedding extraction time",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
)

LIVENESS_MS = Histogram(
    "faceid_liveness_ms",
    "Passive liveness check time",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
)

DB_QUERY_TIME_MS = Histogram(
    "faceid_db_query_time_ms",
    "Database operation duration",
    ["operation"],
    buckets=[0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
)

SEARCH_LATENCY = Histogram(
    "faceid_search_latency_seconds",
    "Search latency",
    buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
)

IS_GENUINE_MODE = Gauge(
    "faceid_is_genuine_mode",
    "Active is_genuine strategy",
    ["mode"],
)
