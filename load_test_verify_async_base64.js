import http from "k6/http";
import { check, sleep } from "k6";
import { Trend } from "k6/metrics";
import encoding from "k6/encoding";

export const options = {
  scenarios: {
    async_base64: {
      executor: "constant-vus",
      vus: Number(__ENV.VUS || 2),
      duration: __ENV.DURATION || "30s",
    },
  },
};

const BASE_URL = "http://localhost:8000";
const IMAGE_FILE = open("./tests/data/person1_small.jpg", "b");
const IMAGE_B64 = encoding.b64encode(IMAGE_FILE);

const queueDelay = new Trend("queue_delay_ms");
const e2eLatency = new Trend("e2e_latency");
const ITERATION_PAUSE = Number(__ENV.PAUSE || 0.1);

function sleepIfNeeded() {
  if (ITERATION_PAUSE > 0) {
    sleep(ITERATION_PAUSE);
  }
}

export default function () {
  const start = Date.now();

  const enqueueRes = http.post(
    `${BASE_URL}/verify_async_base64`,
    JSON.stringify({
      user_id: "1",
      image: IMAGE_B64,
      require_liveness: false,
    }),
    {
      headers: { "Content-Type": "application/json" },
      timeout: "10s",
    }
  );

  check(enqueueRes, {
    "enqueue status 200/202": (r) => r.status === 200 || r.status === 202,
  });

  let jobId = null;
  try {
    jobId = enqueueRes.json("job_id");
  } catch (_) {
    sleepIfNeeded();
    return;
  }

  if (!jobId) {
    sleepIfNeeded();
    return;
  }

  for (let attempts = 0; attempts < 40; attempts++) {
    const resultRes = http.get(`${BASE_URL}/verify_result/${jobId}`, {
      timeout: "10s",
    });

    check(resultRes, {
      "result status 200": (r) => r.status === 200,
    });

    let data = null;
    try {
      data = resultRes.json();
    } catch (_) {
      break;
    }

    if (data && data.ready === true) {
      e2eLatency.add(Date.now() - start);
      if (typeof data.queue_delay_ms === "number") {
        queueDelay.add(data.queue_delay_ms);
      }

      check(data, {
        "job done or failed": (d) => d.status === "done" || d.status === "failed",
      });

      return;
    }

    sleep(0.01);
  }

  sleepIfNeeded();
}
