import http from "k6/http";
import { check, sleep } from "k6";
import { Trend } from "k6/metrics";

export const options = {
  scenarios: {
    async_verify: {
      executor: "ramping-vus",
      startVUs: 1,
      stages: [
        { duration: "30s", target: 10 },
        { duration: "1m", target: 20 },
        { duration: "1m", target: 30 },
        { duration: "30s", target: 0 },
      ],
    },
  },
};

const BASE_URL = "http://localhost:8000";
const IMAGE_FILE = open("./tests/data/person1_small.jpg", "b");

const e2eLatency = new Trend("e2e_latency");

export default function () {
  const start = Date.now();

  const enqueueRes = http.post(
    `${BASE_URL}/verify_async?user_id=1`,
    {
      file: http.file(IMAGE_FILE, "person1_small.jpg", "image/jpeg"),
    },
    { timeout: "10s" }
  );

  check(enqueueRes, {
    "enqueue status 200/202": (r) => r.status === 200 || r.status === 202,
  });

  let jobId = null;
  try {
    jobId = enqueueRes.json("job_id");
  } catch (_) {
    return;
  }

  if (!jobId) return;

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

      check(data, {
        "job done or failed": (d) => d.status === "done" || d.status === "failed",
      });

      return;
    }

    sleep(0.2);
  }
}
