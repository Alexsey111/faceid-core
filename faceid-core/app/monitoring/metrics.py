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

SEARCH_LATENCY = Histogram(
    "faceid_search_latency_seconds",
    "Search latency",
    buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
)
