import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Rate, Trend } from "k6/metrics";
import encoding from "k6/encoding";

export const options = {
  summaryTrendStats: ["avg", "min", "med", "p(90)", "p(95)", "p(99)", "max"],
  scenarios: {
    async_verify: {
      executor: "constant-vus",
      vus: Number(__ENV.VUS || 4),
      duration: __ENV.DURATION || "2m",
      gracefulStop: __ENV.GRACEFUL_STOP || "2m",
    },
  },
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";
const IMAGE_FILE = __ENV.IMAGE_FILE || "./tests/data_extended/person_011/1.jpg";
const NORMAL_IMAGE_FILES = ((__ENV.NORMAL_IMAGE_FILES || IMAGE_FILE) + "")
  .split(",")
  .map((item) => item.trim())
  .filter(Boolean);
const HARD_IMAGE_FILES = ((__ENV.HARD_IMAGE_FILES || __ENV.HARD_IMAGE_FILE || "") + "")
  .split(",")
  .map((item) => item.trim())
  .filter(Boolean);
const HARD_RATIO = Number(__ENV.HARD_RATIO || 0.3);

function loadImageBase64(path) {
  const normalized = path.toLowerCase();

  if (normalized.endsWith(".b64.txt")) {
    return open(path).replace(/^\uFEFF/, "").trim();
  }

  const bytes = open(path, "b");
  return encoding.b64encode(bytes);
}

const NORMAL_IMAGE_BASE64S = NORMAL_IMAGE_FILES.map(loadImageBase64);
const HARD_IMAGE_BASE64S = HARD_IMAGE_FILES.map(loadImageBase64);

if (!NORMAL_IMAGE_BASE64S.length) {
  throw new Error("NORMAL_IMAGE_FILES or IMAGE_FILE is required");
}

const e2eLatency = new Trend("e2e_latency_ms");
const queueDelay = new Trend("queue_delay_ms");
const processingTime = new Trend("processing_time_ms");

const rejectRate = new Rate("reject_rate");
const enqueuePassRate = new Rate("enqueue_pass_rate");
const resultPassRate = new Rate("result_pass_rate");
const completedEventually = new Rate("completed_eventually");

const totalRequests = new Counter("total_requests");
const responses429 = new Counter("responses_429");
const enqueuePasses = new Counter("enqueue_passes");
const enqueueFails = new Counter("enqueue_fails");
const resultPasses = new Counter("result_passes");
const resultFails = new Counter("result_fails");
const terminalExpired = new Counter("terminal_expired");
const waitTimeouts = new Counter("wait_timeouts");
const fastOnlyJobs = new Counter("fast_only_jobs");
const retinafaceJobs = new Counter("retinaface_jobs");

const ITERATION_PAUSE = Number(__ENV.PAUSE || 0.5);
const MAX_POLLS = Number(__ENV.MAX_POLLS || 3);
const WAIT_TIMEOUT_MS = Number(__ENV.WAIT_TIMEOUT_MS || 30000);

function asMsTopLevelOrNested(data, topLevelKey, nestedKey) {
  const topLevel = data?.[topLevelKey];
  if (typeof topLevel === "number" && Number.isFinite(topLevel)) {
    return topLevel;
  }

  const nested = data?.metrics?.[nestedKey];
  if (typeof nested === "number" && Number.isFinite(nested)) {
    return nested * 1000.0;
  }

  if (typeof nested === "string") {
    const parsed = Number(nested);
    if (Number.isFinite(parsed)) {
      return parsed * 1000.0;
    }
  }

  return null;
}

function isTerminalStatus(status) {
  return (
    status === "done" ||
    status === "error" ||
    status === "failed" ||
    status === "expired"
  );
}

function counterValue(data, key, fallback = 0) {
  const metric = data?.metrics?.[key];
  return typeof metric?.values?.count === "number" ? metric.values.count : fallback;
}

function rateValue(data, key, fallback = 0) {
  const metric = data?.metrics?.[key];
  return typeof metric?.values?.rate === "number" ? metric.values.rate : fallback;
}

function trendStats(data, key) {
  const metric = data?.metrics?.[key];
  if (!metric || !metric.values) {
    return null;
  }

  return {
    avg: metric.values.avg ?? null,
    min: metric.values.min ?? null,
    med: metric.values.med ?? null,
    p90: metric.values["p(90)"] ?? null,
    p95: metric.values["p(95)"] ?? null,
    p99: metric.values["p(99)"] ?? null,
    max: metric.values.max ?? null,
  };
}

function bboxSource(data) {
  const resultObj = data?.result ?? data ?? {};
  return resultObj?.bbox_source ?? null;
}

function bboxSourceDetail(data) {
  const resultObj = data?.result ?? data ?? {};
  return resultObj?.bbox_source_detail ?? null;
}

function pickImageBase64() {
  if (HARD_IMAGE_BASE64S.length > 0 && HARD_RATIO > 0 && Math.random() < HARD_RATIO) {
    return HARD_IMAGE_BASE64S[Math.floor(Math.random() * HARD_IMAGE_BASE64S.length)];
  }

  return NORMAL_IMAGE_BASE64S[Math.floor(Math.random() * NORMAL_IMAGE_BASE64S.length)];
}

export default function () {
  const imageBase64 = pickImageBase64();

  const enqueueRes = http.post(
    `${BASE_URL}/verify_async`,
    JSON.stringify({
      image_b64: imageBase64,
      user_id: "1",
      require_liveness: false,
    }),
    {
      headers: { "Content-Type": "application/json" },
      timeout: "10s",
    }
  );

  totalRequests.add(1);

  if (enqueueRes.status === 429) {
    responses429.add(1);
  }

  const enqueueOk = enqueueRes.status === 200 || enqueueRes.status === 202;
  enqueuePassRate.add(enqueueOk ? 1 : 0);
  rejectRate.add(enqueueRes.status === 429 ? 1 : 0);

  if (enqueueOk) {
    enqueuePasses.add(1);
  } else {
    enqueueFails.add(1);
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

  for (let attempts = 0; attempts < MAX_POLLS; attempts++) {
    const resultRes = http.get(
      `${BASE_URL}/jobs/${jobId}/wait?timeout=${WAIT_TIMEOUT_MS}`,
      { timeout: `${WAIT_TIMEOUT_MS + 5000}ms` }
    );

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
      resultFails.add(1);
      resultPassRate.add(0);
      break;
    }

    if (data && isTerminalStatus(data.status)) {
      const resultObj = data.result ?? data;
      const source = resultObj?.bbox_source ?? null;
      const sourceDetail = resultObj?.bbox_source_detail ?? null;

      if (source === "fast") {
        fastOnlyJobs.add(1);
      } else if (source && source.includes("fallback")) {
        retinafaceJobs.add(1);
      }

      const queueDelayValue = asMsTopLevelOrNested(
        data,
        "async_job_queue_delay_ms",
        "queue_delay"
      );
      const processingTimeValue = asMsTopLevelOrNested(
        data,
        "async_job_processing_ms",
        "processing_time"
      );
      const totalLatencyValue = asMsTopLevelOrNested(
        data,
        "async_job_total_latency_ms",
        "total_latency"
      );

      if (queueDelayValue !== null) {
        queueDelay.add(queueDelayValue);
      }
      if (processingTimeValue !== null) {
        processingTime.add(processingTimeValue);
      }
      if (totalLatencyValue !== null) {
        e2eLatency.add(totalLatencyValue);
      }

      check(data, {
        "job reached terminal status": () => true,
      });

      if (data.status === "expired") {
        terminalExpired.add(1);
      }

      resultPasses.add(1);
      resultPassRate.add(1);
      completedEventually.add(1);
      return;
    }

    if (attempts < MAX_POLLS - 1) {
      sleep(0.2);
    }
  }

  waitTimeouts.add(1);
  resultFails.add(1);
  resultPassRate.add(0);
  completedEventually.add(0);

  if (ITERATION_PAUSE > 0) {
    sleep(ITERATION_PAUSE);
  }
}

export function handleSummary(data) {
  const summary = {
    run: {
      base_url: BASE_URL,
      vus: Number(__ENV.VUS || 4),
      duration: __ENV.DURATION || "2m",
      wait_timeout_ms: WAIT_TIMEOUT_MS,
      max_polls: MAX_POLLS,
      pause: ITERATION_PAUSE,
      image_file: IMAGE_FILE,
      normal_image_files: NORMAL_IMAGE_FILES,
      hard_image_files: HARD_IMAGE_FILES,
      hard_ratio: HARD_RATIO,
    },
    counters: {
      total_requests: counterValue(data, "total_requests", 0),
      responses_429: counterValue(data, "responses_429", 0),
      enqueue_passes: counterValue(data, "enqueue_passes", 0),
      enqueue_fails: counterValue(data, "enqueue_fails", 0),
      result_passes: counterValue(data, "result_passes", 0),
      result_fails: counterValue(data, "result_fails", 0),
      terminal_expired: counterValue(data, "terminal_expired", 0),
      wait_timeouts: counterValue(data, "wait_timeouts", 0),
      fast_only_jobs: counterValue(data, "fast_only_jobs", 0),
      retinaface_jobs: counterValue(data, "retinaface_jobs", 0),
    },
    rates: {
      reject_rate: rateValue(data, "reject_rate", 0),
      enqueue_pass_rate: rateValue(data, "enqueue_pass_rate", 0),
      result_pass_rate: rateValue(data, "result_pass_rate", 0),
      completed_eventually: rateValue(data, "completed_eventually", 0),
      http_req_failed: rateValue(data, "http_req_failed", 0),
    },
    trends: {
      http_req_duration_ms: trendStats(data, "http_req_duration"),
      e2e_latency_ms: trendStats(data, "e2e_latency_ms"),
      queue_delay_ms: trendStats(data, "queue_delay_ms"),
      processing_time_ms: trendStats(data, "processing_time_ms"),
      iteration_duration_ms: trendStats(data, "iteration_duration"),
    },
  };

  const compactText = [
    "=== FaceID Async Benchmark Summary ===",
    `VUS=${summary.run.vus} DURATION=${summary.run.duration}`,
    `enqueue_passes=${summary.counters.enqueue_passes} enqueue_fails=${summary.counters.enqueue_fails}`,
    `result_passes=${summary.counters.result_passes} result_fails=${summary.counters.result_fails}`,
    `reject_rate=${summary.rates.reject_rate}`,
    `http_req_failed=${summary.rates.http_req_failed}`,
    `responses_429=${summary.counters.responses_429}`,
    `fast_only_jobs=${summary.counters.fast_only_jobs}`,
    `retinaface_jobs=${summary.counters.retinaface_jobs}`,
    `completed_eventually=${summary.rates.completed_eventually}`,
    `queue_delay_p95_ms=${summary.trends.queue_delay_ms?.p95}`,
    `processing_time_p95_ms=${summary.trends.processing_time_ms?.p95}`,
    `e2e_latency_p95_ms=${summary.trends.e2e_latency_ms?.p95}`,
    `e2e_latency_p99_ms=${summary.trends.e2e_latency_ms?.p99}`,
    "",
  ].join("\n");

  return {
    stdout: compactText,
    "summary.json": JSON.stringify(summary, null, 2),
  };
}
