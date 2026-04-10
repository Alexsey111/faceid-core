import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Trend, Rate } from "k6/metrics";
import encoding from "k6/encoding";

const completed = new Counter("completed");
const waitTimeouts = new Counter("wait_timeouts");
const completionFailed = new Rate("completion_failed");
const totalE2E = new Trend("client_e2e_ms");
const clientIteration = new Trend("client_iteration_ms");
const queueDelay = new Trend("queue_delay_ms");
const processingTime = new Trend("processing_time_ms");
const completedEventually = new Rate("completed_eventually");
const resultPasses = new Counter("result_passes");
const resultFails = new Counter("result_fails");
const enqueueAccepted = new Counter("enqueue_accepted");
const enqueue429 = new Counter("enqueue_429");
const terminalDone = new Counter("terminal_done");
const terminalError = new Counter("terminal_error");
const terminalExpired = new Counter("terminal_expired");
const nonTerminalWait = new Counter("non_terminal_wait");
const waitBadStatus = new Counter("wait_bad_status");

const BASE_URL = __ENV.BASE_URL || "http://localhost:8080";
const RATE = Number(__ENV.RATE || 20);
const DURATION = __ENV.DURATION || "2m";
const RAW_WAIT_TIMEOUT_MS = Number(__ENV.WAIT_TIMEOUT_MS || 30000);
const WAIT_TIMEOUT_MS = RAW_WAIT_TIMEOUT_MS;
if (WAIT_TIMEOUT_MS > 30000) {
  throw new Error(
    `WAIT_TIMEOUT_MS=${WAIT_TIMEOUT_MS} exceeds /jobs/{id}/wait max timeout=30000`,
  );
}
const PRE_ALLOCATED_VUS = Number(__ENV.PRE_ALLOCATED_VUS || __ENV.PREALLOCATED_VUS || 250);
const MAX_VUS = Number(__ENV.MAX_VUS || 500);
const ITERATION_PAUSE = Number(__ENV.PAUSE || 0);
const REQUEST_TIMEOUT = __ENV.REQUEST_TIMEOUT || "30s";

function loadImageBase64(path) {
  if (path.endsWith(".b64.txt")) {
    return open(path).replace(/^\uFEFF/, "").trim();
  }

  return encoding.b64encode(open(path, "b"));
}

const IMAGE_B64 = __ENV.IMAGE_B64
  ? __ENV.IMAGE_B64
  : __ENV.IMAGE_FILE
    ? loadImageBase64(__ENV.IMAGE_FILE)
    : open("./tests/data/person1_small.b64.txt").trim();
const REQUIRE_LIVENESS = (__ENV.REQUIRE_LIVENESS || "false") === "true";

export const options = {
  scenarios: {
    completion_flow: {
      executor: "constant-arrival-rate",
      rate: RATE,
      timeUnit: "1s",
      duration: DURATION,
      preAllocatedVUs: PRE_ALLOCATED_VUS,
      maxVUs: MAX_VUS,
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.05"],
    completion_failed: ["rate<0.05"],
  },
};

function payload() {
  return JSON.stringify({
    user_id: "ext_test_user",
    image_b64: IMAGE_B64,
    require_liveness: REQUIRE_LIVENESS,
  });
}

function isTerminalStatus(status) {
  return status === "done" || status === "error" || status === "failed" || status === "expired";
}

export function setup() {
  const res = http.post(`${BASE_URL}/verify_async`, payload(), {
    headers: { "Content-Type": "application/json" },
    timeout: REQUEST_TIMEOUT,
  });

  if (![200, 202, 429].includes(res.status)) {
    throw new Error(
      `Smoke failed: status=${res.status}, body=${String(res.body).slice(0, 500)}`
    );
  }
}

export default function () {
  const start = Date.now();

  const enqueueRes = http.post(`${BASE_URL}/verify_async`, payload(), {
    headers: { "Content-Type": "application/json" },
    timeout: REQUEST_TIMEOUT,
  });

  const enqueueOk = check(enqueueRes, {
    "enqueue status is 200/202/429": (r) => [200, 202, 429].includes(r.status),
  });

  if (!enqueueOk) {
    if (__ITER < 3) {
      console.error(`bad status=${enqueueRes.status} body=${String(enqueueRes.body).slice(0, 500)}`);
    }
    clientIteration.add(Date.now() - start);
    completionFailed.add(1);
    resultFails.add(1);
    return;
  }

  if (enqueueRes.status === 429) {
    enqueue429.add(1);
    clientIteration.add(Date.now() - start);
    completionFailed.add(0);
    resultFails.add(1);
    return;
  }

  enqueueAccepted.add(1);

  let body = null;
  try {
    body = enqueueRes.json();
  } catch (_) {
    if (__ITER < 3) {
      console.error(`bad json body=${String(enqueueRes.body).slice(0, 500)}`);
    }
    clientIteration.add(Date.now() - start);
    completionFailed.add(1);
    resultFails.add(1);
    return;
  }

  const jobId = body?.job_id;
  if (!jobId) {
    clientIteration.add(Date.now() - start);
    completionFailed.add(1);
    resultFails.add(1);
    return;
  }

  const waitRes = http.get(`${BASE_URL}/jobs/${jobId}/wait?timeout=${WAIT_TIMEOUT_MS}`, {
    timeout: `${WAIT_TIMEOUT_MS + 5000}ms`,
  });

  check(waitRes, {
    "wait status 200": (r) => r.status === 200,
  });

  if (waitRes.status !== 200) {
    waitBadStatus.add(1);

    if (__ITER < 3) {
      console.error(
        `wait bad status=${waitRes.status} body=${String(waitRes.body).slice(0, 500)}`
      );
    }

    clientIteration.add(Date.now() - start);
    completedEventually.add(0);
    completionFailed.add(1);
    resultFails.add(1);
    return;
  }

  let json = null;
  try {
    json = waitRes.json();
  } catch (_) {
    json = null;
  }

  const status = json?.status || "processing";
  const completedAtMs = isTerminalStatus(status) ? Date.now() - start : null;

  if (status === "done") {
    check(json, {
      "has result": (r) => r.result !== undefined,
    });
  } else if (status === "error" || status === "failed") {
    check(json, {
      "has error": (r) => r.error !== undefined,
    });
  }

  const queueDelayMs =
    json?.async_job_queue_delay_ms ?? json?.metrics?.queue_delay ?? null;
  const processingTimeMs =
    json?.async_job_processing_ms ?? json?.metrics?.processing_time ?? null;
  const totalLatencyMs =
    json?.async_job_total_latency_ms ?? json?.metrics?.total_latency ?? null;

  if (typeof queueDelayMs === "number" && Number.isFinite(queueDelayMs)) {
    queueDelay.add(queueDelayMs);
  }
  if (typeof processingTimeMs === "number" && Number.isFinite(processingTimeMs)) {
    processingTime.add(processingTimeMs);
  }
  if (typeof totalLatencyMs === "number" && Number.isFinite(totalLatencyMs)) {
    totalE2E.add(totalLatencyMs);
  }

  if (completedAtMs !== null) {
    completed.add(1);
    completedEventually.add(1);
    resultPasses.add(1);
    completionFailed.add(0);
    if (status === "done") {
      terminalDone.add(1);
    } else if (status === "error" || status === "failed") {
      terminalError.add(1);
    } else if (status === "expired") {
      terminalExpired.add(1);
    }
  } else {
    waitTimeouts.add(1);
    nonTerminalWait.add(1);
    completedEventually.add(0);
    resultFails.add(1);
    completionFailed.add(1);
  }

  clientIteration.add(Date.now() - start);

  if (ITERATION_PAUSE > 0) {
    sleep(ITERATION_PAUSE);
  }
}
