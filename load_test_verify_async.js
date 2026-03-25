import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Trend } from "k6/metrics";

export const options = {
  scenarios: {
    async_verify: {
      executor: "constant-arrival-rate",
      rate: Number(__ENV.RATE || 5),
      timeUnit: "1s",
      preAllocatedVUs: Number(__ENV.PREALLOCATED_VUS || 5),
      maxVUs: Number(__ENV.MAX_VUS || 20),
      duration: __ENV.DURATION || "2m",
    },
  },
};

const BASE_URL = "http://localhost:8000";
const IMAGE_FILE = open("./tests/data/person1_small.jpg", "b");

const e2eLatency = new Trend("e2e_latency");
const queueDelay = new Trend("queue_delay_ms");
const totalRequests = new Counter("total_requests");
const responses429 = new Counter("responses_429");
const ITERATION_PAUSE = Number(__ENV.PAUSE || 0.5);

export default function () {
  const start = Date.now();

  const enqueueRes = http.post(
    `${BASE_URL}/verify_async?user_id=1&priority=low`,
    {
      file: http.file(IMAGE_FILE, "person1_small.jpg", "image/jpeg"),
    },
    { timeout: "10s" }
  );
  totalRequests.add(1);
  if (enqueueRes.status === 429) {
    responses429.add(1);
  }

  check(enqueueRes, {
    "enqueue status 200/202": (r) => r.status === 200 || r.status === 202,
  });

  let jobId = null;
  try {
    jobId = enqueueRes.json("job_id");
  } catch (_) {
    if (ITERATION_PAUSE > 0) {
      sleep(ITERATION_PAUSE);
    }
    return;
  }

  if (!jobId) {
    if (ITERATION_PAUSE > 0) {
      sleep(ITERATION_PAUSE);
    }
    return;
  }

  for (let attempts = 0; attempts < 120; attempts++) {
    const resultRes = http.get(`${BASE_URL}/verify_result/${jobId}`, {
      timeout: "10s",
    });
    totalRequests.add(1);
    if (resultRes.status === 429) {
      responses429.add(1);
    }

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
      let queueDelayValue = data.queue_delay_ms;
      if (typeof queueDelayValue !== "number") {
        for (let extra = 0; extra < 20; extra++) {
          sleep(0.1);
          const cachedRes = http.get(`${BASE_URL}/verify_result/${jobId}`, {
            timeout: "10s",
          });
          totalRequests.add(1);
          if (cachedRes.status === 429) {
            responses429.add(1);
          }

          try {
            const cachedData = cachedRes.json();
            if (cachedData && typeof cachedData.queue_delay_ms === "number") {
              queueDelayValue = cachedData.queue_delay_ms;
              break;
            }
          } catch (_) {
          }
        }
      }

      if (typeof queueDelayValue === "number") {
        queueDelay.add(queueDelayValue);
      }

      check(data, {
        "job done or failed": (d) => d.status === "done" || d.status === "failed",
      });

      return;
    }

    sleep(0.1);
  }

  if (ITERATION_PAUSE > 0) {
    sleep(ITERATION_PAUSE);
  }
}
