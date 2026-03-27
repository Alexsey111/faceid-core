import http from "k6/http";
import { check } from "k6";

export const options = {
  scenarios: {
    async_verify: {
      executor: "constant-arrival-rate",
      rate: Number(__ENV.RATE || 20),
      timeUnit: "1s",
      duration: __ENV.DURATION || "3m",
      preAllocatedVUs: Number(__ENV.PREALLOCATED_VUS || 50),
      maxVUs: Number(__ENV.MAX_VUS || 100),
    },
  },
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";
const IMAGE_BASE64 = (__ENV.IMAGE || open(__ENV.IMAGE_FILE || "./tests/data/person1_small.b64.txt")).trim();
const SLA_FAST_MS = Number(__ENV.SLA_FAST_MS || 1500);
const SLA_ASYNC_MS = Number(__ENV.SLA_ASYNC_MS || 5000);
const MAX_WAIT_MS = Number(__ENV.MAX_WAIT_MS || 25000);

export default function () {
  const enqueueRes = http.post(
    `${BASE_URL}/verify_async`,
    JSON.stringify({
      image_b64: IMAGE_BASE64,
    }),
    {
      headers: { "Content-Type": "application/json" },
      timeout: "10s",
    }
  );

  check(enqueueRes, {
    "enqueue_success": (r) => r.status === 200,
  });

  let body = null;
  try {
    body = enqueueRes.json();
  } catch (_) {
    return;
  }

  const jobId = body && body.job_id;
  if (!jobId) {
    return;
  }

  const start = Date.now();
  const waitRes = http.get(`${BASE_URL}/jobs/${jobId}/wait?timeout=2000`, {
    timeout: "10s",
  });

  check(waitRes, {
    "wait status 200": (r) => r.status === 200,
  });

  let json = null;
  try {
    json = waitRes.json();
  } catch (_) {
    json = null;
  }

  const status = json && json.status ? json.status : "processing";
  const completedAtMs =
    status === "done" || status === "error" || status === "failed"
      ? Date.now() - start
      : null;

  if (status === "done") {
    check(json, {
      "has result": (r) => r.result !== undefined,
    });
  } else if (status === "error" || status === "failed") {
    check(json, {
      "has error": (r) => r.error !== undefined,
    });
  }

  check({ completedAtMs }, {
    "completed_within_fast_sla": (r) => r.completedAtMs !== null && r.completedAtMs <= SLA_FAST_MS,
    "completed_within_async_sla": (r) => r.completedAtMs !== null && r.completedAtMs <= SLA_ASYNC_MS,
    "completed_eventually": (r) => r.completedAtMs !== null,
  });
}
