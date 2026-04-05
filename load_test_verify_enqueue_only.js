import http from "k6/http";
import { check } from "k6";
import { Counter, Rate } from "k6/metrics";

const enqueueAccepted = new Counter("enqueue_accepted");
const enqueueRejected = new Counter("enqueue_rejected");
const enqueueFailed = new Rate("enqueue_failed");

const BASE_URL = __ENV.BASE_URL || "http://localhost:8080";
const RATE = Number(__ENV.RATE || 30);
const DURATION = __ENV.DURATION || "2m";
const PRE_ALLOCATED_VUS = Number(__ENV.PRE_ALLOCATED_VUS || __ENV.PREALLOCATED_VUS || 200);
const MAX_VUS = Number(__ENV.MAX_VUS || 400);
const REQUEST_TIMEOUT = __ENV.REQUEST_TIMEOUT || "30s";

const IMAGE_B64 = __ENV.IMAGE_B64
  ? __ENV.IMAGE_B64
  : __ENV.IMAGE_FILE
    ? open(__ENV.IMAGE_FILE).trim()
    : open("./tests/data/person1_small.b64.txt").trim();

export const options = {
  scenarios: {
    enqueue_only: {
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
    enqueue_failed: ["rate<0.05"],
  },
};

function buildPayload() {
  return JSON.stringify({
    user_id: "ext_test_user",
    image_b64: IMAGE_B64,
    require_liveness: false,
  });
}

export function setup() {
  const res = http.post(`${BASE_URL}/verify_async`, buildPayload(), {
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
  const res = http.post(`${BASE_URL}/verify_async`, buildPayload(), {
    headers: { "Content-Type": "application/json" },
    timeout: REQUEST_TIMEOUT,
  });

  const ok = check(res, {
    "enqueue status is 200/202/429": (r) => [200, 202, 429].includes(r.status),
  });

  if (!ok) {
    if (__ITER < 3) {
      console.error(`bad status=${res.status} body=${String(res.body).slice(0, 500)}`);
    }
    enqueueFailed.add(true);
    return;
  }

  if (res.status === 200 || res.status === 202) {
    enqueueAccepted.add(1);
    enqueueFailed.add(false);
    return;
  }

  if (res.status === 429) {
    enqueueRejected.add(1);
    enqueueFailed.add(false);
    return;
  }

  enqueueFailed.add(true);
}
