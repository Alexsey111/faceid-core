import http from "k6/http";
import { check, sleep } from "k6";
import { Trend } from "k6/metrics";
import encoding from "k6/encoding";

export const options = {
  scenarios: {
    direct_verify: {
      executor: "constant-vus",
      vus: Number(__ENV.VUS || 5),
      duration: __ENV.DURATION || "30s",
    },
  },
};

const BASE_URL = "http://localhost:8000";
const IMAGE_FILE = open("./tests/data/person1_small.jpg", "b");
const IMAGE_B64 = encoding.b64encode(IMAGE_FILE);

const e2eLatency = new Trend("e2e_latency");
const ITERATION_PAUSE = Number(__ENV.PAUSE || 0.5);

export default function () {
  const start = Date.now();

  const verifyRes = http.post(
    `${BASE_URL}/verify_base64`,
    JSON.stringify({
      user_id: "1",
      image: IMAGE_B64,
    }),
    {
      headers: { "Content-Type": "application/json" },
      timeout: "10s",
    }
  );

  check(verifyRes, {
    "verify status 200": (r) => r.status === 200,
  });

  e2eLatency.add(Date.now() - start);

  if (ITERATION_PAUSE > 0) {
    sleep(ITERATION_PAUSE);
  }
}
