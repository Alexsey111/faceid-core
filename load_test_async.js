import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  vus: Number(__ENV.VUS || 10),
  duration: __ENV.DURATION || "30s",
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";
const IMAGE_BASE64 = (__ENV.IMAGE || open(__ENV.IMAGE_FILE || "./tests/data/person1_small.b64.txt")).trim();
const MAX_WAIT_MS = Number(__ENV.MAX_WAIT_MS || 5000);
const POLL_INTERVAL_MS = Number(__ENV.POLL_INTERVAL_MS || 100);

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
    "enqueue status 200": (r) => r.status === 200,
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
  let status = "processing";

  while (status === "processing" && Date.now() - start < MAX_WAIT_MS) {
    const statusRes = http.get(`${BASE_URL}/jobs/${jobId}`, {
      timeout: "10s",
    });

    check(statusRes, {
      "status status 200": (r) => r.status === 200,
    });

    let json = null;
    try {
      json = statusRes.json();
    } catch (_) {
      break;
    }

    status = json && json.status ? json.status : "processing";

    if (status === "done") {
      check(json, {
        "has result": (r) => r.result !== undefined,
      });
      return;
    }

    if (status === "error" || status === "failed") {
      check(json, {
        "has error": (r) => r.error !== undefined,
      });
      return;
    }

    sleep(POLL_INTERVAL_MS / 1000);
  }

  check(status, {
    "completed": (s) => s === "done",
  });
}
