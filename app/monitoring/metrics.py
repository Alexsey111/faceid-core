# app\monitoring\metrics.py

from typing import Any

from prometheus_client import Counter, Gauge, Histogram, REGISTRY


def _get_or_create_metric(factory, name: str, *args, **kwargs) -> Any:
    try:
        return factory(name, *args, **kwargs)
    except ValueError:
        collector = REGISTRY._names_to_collectors.get(name)
        if collector is not None:
            return collector
        raise


def _counter(name: str, *args, **kwargs) -> Any:
    return _get_or_create_metric(Counter, name, *args, **kwargs)


def _gauge(name: str, *args, **kwargs) -> Any:
    return _get_or_create_metric(Gauge, name, *args, **kwargs)


def _histogram(name: str, *args, **kwargs) -> Any:
    return _get_or_create_metric(Histogram, name, *args, **kwargs)


REQUEST_COUNTER = _counter(
    "faceid_http_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"],
)

FAISS_HIT = _counter(
    "faceid_faiss_hit_total",
    "FAISS hits",
    ["endpoint", "result"],
)

REDIS_HIT = _counter(
    "faceid_redis_hit_total",
    "Redis cache hits",
    ["endpoint", "result"],
)

DB_FALLBACK = _counter(
    "faceid_db_fallback_total",
    "Fallback to DB",
    ["endpoint", "result"],
)

ERROR_COUNTER = _counter(
    "faceid_errors_total",
    "Application errors",
    ["stage", "error_type"],
)

VERIFICATION_RESULT_COUNTER = _counter(
    "faceid_verification_result_total",
    "Verification outcomes",
    ["status", "liveness_passed"],
)

VERIFY_RESULT = _counter(
    "faceid_verify_result_total",
    "Verification results",
    ["result"],
)

VERIFY_RESULT_COUNTER = VERIFY_RESULT

LIVENESS_RESULT_COUNTER = _counter(
    "faceid_liveness_result_total",
    "Liveness outcomes",
    ["result"],
)

LIVENESS_FAIL_COUNT = _counter(
    "faceid_liveness_fail_total",
    "Failed liveness checks",
)

QUALITY_REJECT_COUNTER = _counter(
    "faceid_quality_reject_total",
    "Quality gate rejects",
    ["reason"],
)

SEARCH_BACKEND_COUNTER = _counter(
    "faceid_search_backend_total",
    "Search backend usage",
    ["backend"],
)

REQUEST_LATENCY = _histogram(
    "faceid_http_request_duration_seconds",
    "API request latency",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0],
)

VERIFY_LATENCY = _histogram(
    "faceid_verify_duration_seconds",
    "Full verify pipeline latency",
)

INPROGRESS_REQUESTS = _gauge(
    "faceid_http_inprogress_requests",
    "HTTP requests currently in progress",
)

PIPELINE_STAGE_DURATION = _histogram(
    "faceid_pipeline_stage_duration_seconds",
    "Verification pipeline stage duration",
    ["stage"],
    buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
)

ALIGN_CROP_MS = _histogram(
    "faceid_align_crop_ms",
    "Face alignment and crop time",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
)

QUEUE_DELAY_MS = _histogram(
    "faceid_queue_delay_ms",
    "Queue delay before worker starts",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
)

PREPROCESS_MS = _histogram(
    "faceid_preprocess_ms",
    "Image preprocessing time",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
)

ASYNC_JOB_TOTAL_LATENCY_MS = _histogram(
    "faceid_async_job_total_latency_ms",
    "Async verification job total latency",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
)

ASYNC_JOB_E2E_LATENCY_MS = _histogram(
    "faceid_job_e2e_latency_ms",
    "Async verification job end-to-end latency",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
)

ASYNC_JOB_QUEUE_DELAY_MS = _histogram(
    "faceid_async_job_queue_delay_ms",
    "Async verification job queue delay",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
)

ASYNC_JOB_PROCESSING_MS = _histogram(
    "faceid_async_job_processing_ms",
    "Async verification job processing time",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
)

ASYNC_JOB_COMPLETED_TOTAL = _counter(
    "faceid_async_job_completed_total",
    "Completed async verification jobs",
)

ASYNC_JOB_ENQUEUED_TOTAL = _counter(
    "faceid_async_job_enqueued_total",
    "Total async jobs accepted into queue",
)

VERIFY_INFLIGHT_INCREMENT_TOTAL = _counter(
    "faceid_verify_inflight_increment_total",
    "Total inflight increments for accepted async jobs",
    ["reason"],
)

VERIFY_INFLIGHT_DECREMENT_TOTAL = _counter(
    "faceid_verify_inflight_decrement_total",
    "Total inflight decrements when async jobs reach a terminal state",
    ["reason"],
)

ASYNC_JOB_TERMINAL_TOTAL = _counter(
    "faceid_async_job_terminal_total",
    "Total async jobs that reached terminal state",
    ["status"],
)

VERIFY_JOB_TERMINAL_TOTAL = _counter(
    "faceid_verify_job_terminal_total",
    "Terminal async job outcomes",
    ["state"],
)

ASYNC_JOB_EXPIRED_TOTAL = _counter(
    "faceid_async_job_expired_total",
    "Expired async verification jobs",
)

JOB_AGE_MS = _histogram(
    "faceid_job_age_ms",
    "Age of async verification job before worker decision",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000],
)

VERIFY_JOB_AGE_ON_FINALIZE_MS = _histogram(
    "faceid_verify_job_age_on_finalize_ms",
    "Age of async verification job when it reaches terminal finalization",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000],
)

VERIFY_WORKER_FINALIZE_TOTAL = _counter(
    "faceid_verify_worker_finalize_total",
    "Terminal job finalization attempts",
    ["state"],
)

VERIFY_WORKER_FINALIZE_FAIL_TOTAL = _counter(
    "faceid_verify_worker_finalize_fail_total",
    "Failed terminal job finalization attempts",
)

VERIFY_WORKER_CLAIM_TO_RESULT_VISIBLE_MS = _histogram(
    "faceid_verify_worker_claim_to_result_visible_ms",
    "Time from worker claim to result becoming visible",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000],
)

VERIFY_WORKER_CLAIM_TO_FINALIZE_MS = _histogram(
    "faceid_verify_worker_claim_to_finalize_ms",
    "Time from worker claim to terminal finalize completion",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000],
)

VERIFY_WORKER_RESULT_WRITE_MS = _histogram(
    "faceid_verify_worker_result_write_ms",
    "Time spent writing terminal result from the worker",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
)

VERIFY_WORKER_TERMINAL_GAP_MS = _histogram(
    "faceid_verify_worker_terminal_gap_ms",
    "Gap between result becoming visible and finalize completion",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
)

VERIFY_RESULT_WRITE_MS = _histogram(
    "faceid_verify_result_write_ms",
    "Time spent writing terminal result payload to Redis",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
)

VERIFY_RESULT_VISIBLE_TOTAL = _counter(
    "faceid_verify_result_visible_total",
    "Terminal results that became visible in Redis",
    ["status"],
)

VERIFY_RESULT_VISIBLE_LAG_MS = _histogram(
    "faceid_verify_result_visible_lag_ms",
    "Age from job completion to result visibility",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
)

VERIFY_WAIT_REQUEST_TOTAL = _counter(
    "faceid_verify_wait_request_total",
    "Total /wait requests",
)

VERIFY_WAIT_HIT_TOTAL = _counter(
    "faceid_verify_wait_hit_total",
    "Total /wait requests that hit a terminal result",
)

VERIFY_WAIT_MISS_TOTAL = _counter(
    "faceid_verify_wait_miss_total",
    "Total /wait lookup misses while polling",
)

VERIFY_WAIT_TIMEOUT_TOTAL = _counter(
    "faceid_verify_wait_timeout_total",
    "Total /wait requests that timed out before a terminal result",
)

VERIFY_WAIT_EMPTY_CYCLES = _counter(
    "faceid_verify_wait_empty_cycles",
    "Total empty polling cycles while waiting for job results",
)

VERIFY_WAIT_HOLD_MS = _histogram(
    "faceid_verify_wait_hold_ms",
    "Total time spent inside /wait before response",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
)

VERIFY_WAIT_LOOKUP_MS = _histogram(
    "faceid_verify_wait_lookup_ms",
    "Total Redis lookup time spent inside /wait",
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
)

VERIFY_VISIBLE_TO_FIRST_HIT_MS = _histogram(
    "faceid_verify_visible_to_first_hit_ms",
    "How old the result was when /wait observed it",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
)

VERIFY_TERMINAL_GAP_MS = _histogram(
    "faceid_verify_terminal_gap_ms",
    "Gap from result visibility to /wait response or finalization boundary",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
)

PIPELINE_MS = _histogram(
    "faceid_pipeline_ms",
    "Total pipeline processing time",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
)

DETECT_MS = _histogram(
    "faceid_detect_ms",
    "Face detection time",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
)

ENCODE_MS = _histogram(
    "faceid_encode_ms",
    "Face embedding extraction time",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
)

VECTOR_SEARCH_MS = _histogram(
    "faceid_vector_search_ms",
    "Vector search time",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
)

RESULT_WRITE_MS = _histogram(
    "faceid_result_write_ms",
    "Result write time",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
)

LIVENESS_MS = _histogram(
    "faceid_liveness_ms",
    "Passive liveness check time",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
)

QUALITY_GATE_PRE_MS = _histogram(
    "faceid_quality_gate_pre_ms",
    "Pre-detect quality gate latency",
    buckets=[0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
)

QUALITY_GATE_FACE_MS = _histogram(
    "faceid_quality_gate_face_ms",
    "Post-detect quality gate latency",
    buckets=[0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
)

DB_QUERY_TIME_MS = _histogram(
    "faceid_db_query_time_ms",
    "Database operation duration",
    ["operation"],
    buckets=[0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
)

SEARCH_LATENCY = _histogram(
    "faceid_search_latency_seconds",
    "Search latency",
    buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
)

# SQL-fallback поиска после удаления plaintext-колонки: сколько векторов пришлось
# расшифровать (decrypt-all). Основной путь — FAISS; fallback срабатывает при
# отсутствии/падении FAISS. Большое N — деградация, нужен наблюдаемый сигнал.
SEARCH_DECRYPT_ALL_FALLBACK_N = _gauge(
    "faceid_search_decrypt_all_fallback_n",
    "Number of vectors decrypted in SQL-fallback search (decrypt-all path)",
)

REDIS_COMMAND_LATENCY_MS = _histogram(
    "faceid_redis_command_latency_ms",
    "Redis command latency",
    ["command"],
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
)

QUEUE_PUSH_LATENCY_MS = _histogram(
    "faceid_queue_push_latency_ms",
    "Async queue push latency",
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
)

QUEUE_POP_LATENCY_MS = _histogram(
    "faceid_queue_pop_latency_ms",
    "Async queue pop latency",
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
)

QUEUE_BATCH_SIZE = _histogram(
    "faceid_queue_batch_size",
    "Number of jobs collected into one dequeue batch",
    buckets=[1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64],
)

QUEUE_JOBS_PER_POP = _histogram(
    "faceid_queue_jobs_per_pop",
    "Number of jobs obtained per dequeue cycle",
    buckets=[1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64],
)

QUEUE_ASSIGNMENT_DELAY_MS = _histogram(
    "faceid_queue_assignment_delay_ms",
    "Time from enqueue until batch dispatch",
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
)

WORKER_IDLE_GAP_MS = _histogram(
    "faceid_worker_idle_gap_ms",
    "Time worker spends idle before the first job of a batch is collected",
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
)

QUEUE_TIME_TO_FIRST_CLAIM_MS = _histogram(
    "faceid_queue_time_to_first_claim_ms",
    "Time from enqueue until the first job in a batch is claimed",
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
)

QUEUE_CLAIM_TO_BATCH_FILL_MS = _histogram(
    "faceid_queue_claim_to_batch_fill_ms",
    "Time from first claim in a batch until the batch is filled",
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
)

QUEUE_BATCH_READY_TO_PROCESSING_START_MS = _histogram(
    "faceid_queue_batch_ready_to_processing_start_ms",
    "Time from batch fill until worker starts processing the batch",
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
)

QUEUE_ENQUEUE_TO_WORKER_ATTEMPT_MS = _histogram(
    "faceid_queue_enqueue_to_worker_attempt_ms",
    "Time from enqueue until the worker starts attempting to claim the job",
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
)

QUEUE_WORKER_ATTEMPT_TO_CLAIM_SUCCESS_MS = _histogram(
    "faceid_queue_worker_attempt_to_claim_success_ms",
    "Time from worker claim attempt start until Redis returns the claimed job",
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
)

WORKER_ACTIVE_BATCHES = _gauge(
    "faceid_worker_active_batches",
    "Number of batches currently being processed by the worker",
)

QUEUE_CLAIM_ATTEMPTS_TOTAL = _counter(
    "faceid_queue_claim_attempts_total",
    "Total worker claim attempts",
)

QUEUE_CLAIM_SUCCESS_TOTAL = _counter(
    "faceid_queue_claim_success_total",
    "Total successful worker claims",
)

WORKER_SEMAPHORE_WAIT_MS = _histogram(
    "faceid_worker_semaphore_wait_ms",
    "Time waiting to acquire the worker semaphore for batch processing",
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
)

IS_GENUINE_MODE = _gauge(
    "faceid_is_genuine_mode",
    "Active is_genuine strategy",
    ["mode"],
)

VERIFY_ACCEPTED_JOBS = _counter(
    "faceid_verify_accepted_jobs_total",
    "Accepted async verification jobs",
)

VERIFY_REJECTED_JOBS = _counter(
    "faceid_verify_rejected_jobs_total",
    "Rejected async verification jobs",
    ["reason"],
)

RATE_LIMIT_HITS = _counter(
    "faceid_rate_limit_hits_total",
    "Number of rate limit rejections",
)

VERIFY_INFLIGHT_CURRENT = _gauge(
    "faceid_verify_inflight_current",
    "Deprecated: process-local last-seen snapshot of inflight_jobs. "
    "Do not use as a global drain/capacity source of truth under scaled workers.",
)

VERIFY_INFLIGHT_REDIS_SNAPSHOT = _gauge(
    "faceid_verify_inflight_redis_snapshot",
    "Process-local snapshot of Redis key inflight_jobs last observed by this process. "
    "Useful for per-process diagnostics only; not a global truth under scaled workers.",
)

QUEUE_LENGTH_REDIS_SNAPSHOT = _gauge(
    "faceid_queue_jobs_pending_redis_snapshot",
    "Process-local snapshot of Redis LLEN(face_verify_queue) last observed by this process. "
    "Useful for per-process diagnostics only; not a global truth under scaled workers.",
)

VERIFY_WORKER_UTILIZATION = Gauge(
    "faceid_verify_worker_utilization",
    "Async verification worker utilization",
)

QUEUE_LENGTH = Gauge(
    "faceid_queue_jobs_pending",
    "Current verification queue length",
)

VERIFY_ASYNC_STATUS_TOTAL = Counter(
    "faceid_verify_async_status_total",
    "HTTP status codes for POST /verify_async",
    ["status"],
)

VERIFY_ASYNC_ACCEPTED_TOTAL = Counter(
    "faceid_verify_async_accepted_total",
    "Accepted POST /verify_async requests that were queued",
)

VERIFY_ASYNC_REJECTED_TOTAL = Counter(
    "faceid_verify_async_rejected_total",
    "Rejected POST /verify_async requests before queueing",
    ["reason"],
)

VERIFY_ASYNC_HTTP_INFLIGHT = Gauge(
    "faceid_verify_async_http_inflight",
    "Current in-flight POST /verify_async requests",
)

VERIFY_ASYNC_REQUEST_SIZE_BYTES = Histogram(
    "faceid_verify_async_request_size_bytes",
    "HTTP body size for POST /verify_async",
    buckets=[
        10_000,
        50_000,
        100_000,
        250_000,
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
    ],
)

VERIFY_ASYNC_IMAGE_B64_CHARS = Histogram(
    "faceid_verify_async_image_b64_chars",
    "Length of image_b64 string in POST /verify_async",
    buckets=[
        10_000,
        50_000,
        100_000,
        250_000,
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
    ],
)

VERIFY_ASYNC_IMAGE_BYTES = Histogram(
    "faceid_verify_async_image_bytes",
    "Decoded image byte size in POST /verify_async",
    buckets=[
        10_000,
        50_000,
        100_000,
        250_000,
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
    ],
)

VERIFY_ASYNC_ADMISSION_MS = Histogram(
    "faceid_verify_async_admission_ms",
    "Full admission latency for POST /verify_async measured in middleware",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 30000],
)

VERIFY_ASYNC_ROUTE_MS = Histogram(
    "faceid_verify_async_route_ms",
    "Time spent inside verify_async route body",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 30000],
)

VERIFY_ASYNC_BASE64_DECODE_MS = Histogram(
    "faceid_verify_async_base64_decode_ms",
    "Base64 decode time inside verify_async",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
)

VERIFY_ASYNC_IMAGE_DECODE_MS = Histogram(
    "faceid_verify_async_image_decode_ms",
    "OpenCV image decode time inside verify_async",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
)

VERIFY_ASYNC_PRECHECK_MS = _histogram(
    "faceid_verify_async_precheck_ms",
    "Light synchronous image prechecks inside verify_async",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
)

VERIFY_ASYNC_ENQUEUE_MS = _histogram(
    "faceid_verify_async_enqueue_ms",
    "Queue enqueue time inside verify_async",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
)

VERIFY_ASYNC_RESPONSE_BUILD_MS = _histogram(
    "faceid_verify_async_response_build_ms",
    "Response build time inside verify_async",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200],
)

VERIFY_ASYNC_BODY_READ_MS = _histogram(
    "faceid_verify_async_body_read_ms",
    "Time to read raw request body inside verify_async",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
)

VERIFY_ASYNC_JSON_PARSE_MS = _histogram(
    "faceid_verify_async_json_parse_ms",
    "Time to parse JSON body inside verify_async",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
)

VERIFY_ASYNC_MODEL_VALIDATE_MS = _histogram(
    "faceid_verify_async_model_validate_ms",
    "Time to validate parsed payload into VerifyAsyncRequest",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
)

VERIFY_ASYNC_MIDDLEWARE_HITS_TOTAL = _counter(
    "faceid_verify_async_middleware_hits_total",
    "How many times verify_async branch in metrics middleware was triggered",
)

VERIFY_ADMISSION_ATTEMPTS_TOTAL = Counter(
    "faceid_verify_admission_attempts_total",
    "Количество входов в admission path",
    ["route"],
)

VERIFY_ADMISSION_ACCEPTED_TOTAL = Counter(
    "faceid_verify_admission_accepted_total",
    "Количество запросов, прошедших admission",
    ["route"],
)

VERIFY_ADMISSION_REJECTED_TOTAL = Counter(
    "faceid_verify_admission_rejected_total",
    "Количество запросов, отклонённых admission",
    ["route", "reason"],
)

VERIFY_ADMISSION_ERRORS_TOTAL = Counter(
    "faceid_verify_admission_errors_total",
    "Количество ошибок внутри admission path",
    ["route", "stage"],
)

VERIFY_ADMISSION_STAGE_MS = Histogram(
    "faceid_verify_admission_stage_ms",
    "Время стадий admission path",
    ["route", "stage", "outcome"],
)

VERIFY_ADMISSION_INFLIGHT_SNAPSHOT = Histogram(
    "faceid_verify_admission_inflight_snapshot",
    "Значение inflight в момент решения admission",
    ["route", "decision"],
)

VERIFY_ADMISSION_QUEUE_LEN_SNAPSHOT = Histogram(
    "faceid_verify_admission_queue_len_snapshot",
    "Длина очереди в момент решения admission",
    ["route", "decision"],
)

VERIFY_ADMISSION_ESTIMATED_DELAY_MS = Histogram(
    "faceid_verify_admission_estimated_delay_ms",
    "Оценённая задержка в момент решения admission",
    ["route", "decision"],
)

ASYNC_STAGE_LATENCY_MS = _histogram(
    "faceid_async_stage_latency_ms",
    "Latency of async stages in milliseconds",
    ["stage"],
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 350, 500, 750, 1000, 1500, 2500, 5000),
)

PIPELINE_STAGE_LATENCY_MS = _histogram(
    "faceid_pipeline_stage_latency_ms",
    "Latency of pipeline stages in milliseconds",
    ["stage"],
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 350, 500, 750, 1000, 1500, 2500, 5000),
)

ASYNC_STAGE_FAILURES_TOTAL = _counter(
    "faceid_async_stage_failures_total",
    "Failures by async stage",
    ["stage", "reason"],
)


def observe_async_stage(stage: str, value_ms: float) -> None:
    ASYNC_STAGE_LATENCY_MS.labels(stage=stage).observe(value_ms)


def observe_pipeline_stage(stage: str, value_ms: float) -> None:
    PIPELINE_STAGE_LATENCY_MS.labels(stage=stage).observe(value_ms)


def inc_async_stage_failure(stage: str, reason: str) -> None:
    ASYNC_STAGE_FAILURES_TOTAL.labels(stage=stage, reason=reason).inc()


# -------------------------
# Webhook delivery (ТЗ 3.2 — интеграция)
# -------------------------
WEBHOOK_DELIVERY_TOTAL = _counter(
    "faceid_webhook_delivery_total",
    "Webhook delivery attempts by terminal state and HTTP outcome",
    ["state", "status"],  # status: success / retry / failed / dropped
)

WEBHOOK_DELIVERY_FAILED = _counter(
    "faceid_webhook_delivery_failed_total",
    "Webhook delivery failures by reason",
    ["reason"],  # receiver_unavailable / timeout / non_2xx / queue_full / idempotent_skip
)

WEBHOOK_DELIVERY_LATENCY = _histogram(
    "faceid_webhook_delivery_latency_ms",
    "Webhook HTTP POST latency (single attempt)",
    buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
)

WEBHOOK_QUEUE_DEPTH = _gauge(
    "faceid_webhook_queue_depth",
    "Current number of pending webhook deliveries in the in-process queue",
)

# -------------------------
# MinIO cleanup (ТЗ 5: исходные фото не хранятся)
# -------------------------
# Счётчик сбоев явного удаления исходного фото. Страховка — MinIO lifecycle
# (expire verify/ через 1ч, см. minio_client._ensure_bucket), но провалы явного
# delete нужно наблюдать.
MINIO_DELETE_FAIL_TOTAL = _counter(
    "faceid_minio_delete_fail_total",
    "Failures to delete an original verification image from MinIO",
    ["stage"],  # verify_task_success / verify_task_failed
)
