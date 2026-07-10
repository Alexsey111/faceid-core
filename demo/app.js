// app.js — логика демо-GUI FaceID Core. Vanilla JS, без зависимостей.
//
// Назначение: презентация работоспособности сервиса на локальном ПК. Кадры с
// веб-камеры → API (/api/v1/*) → результат. Same-origin на localhost:8000.
//
// БЕЗОПАСНОСТЬ (152-ФЗ): кадры только в памяти (canvas → fetch → обнуляем ссылку).
// Без localStorage/sessionStorage/console.log для base64 и эмбеддингов.
"use strict";

// ---- Состояние ----
const state = {
  stream: null,          // MediaStream камеры
  ws: null,              // WebSocket active challenge
  wsFramesSent: 0,       // счётчик кадров в сессии (лимит 30)
  wsInterval: null,      // setInterval стриминга
  livenessToken: null,   // liveness_token из active challenge (только в памяти)
  config: null,          // кэш /api/v1/config
  matchThreshold: 0.6,   // client-side порог (значение слайдера)
};

// ---- DOM-хелперы ----
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const video   = $("#cam");
const canvas  = $("#canvas");
const ctx     = canvas.getContext("2d");

// ---- API-хелпер (same-origin, AUTH_ENABLED=false → без заголовков) ----
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let body;
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    body = await res.json();
  } else {
    body = await res.text();
  }
  if (!res.ok) {
    // Пробрасываем структурированную ошибку для отображения
    const msg = (body && (body.detail || body.message)) || res.statusText;
    const err = new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    err.status = res.status;
    err.body = body;
    throw err;
  }
  return body;
}

// ---- Камера ----
async function camStart() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    video.srcObject = state.stream;
    await video.play();
    $("#cam-start").disabled = true;
    $("#cam-stop").disabled = false;
    $("#cam-status").textContent = "Камера активна.";
  } catch (e) {
    $("#cam-status").textContent = "Ошибка камеры: " + e.message;
  }
}

function camStop() {
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
    video.srcObject = null;
  }
  $("#cam-start").disabled = false;
  $("#cam-stop").disabled = true;
  $("#cam-status").textContent = "Камера остановлена.";
}

// Возвращает чистый base64 JPEG (без data:-prefix) для /verify_base64, /upload_base64.
// Кадр только в памяти; вызывающий код не должен его персистить.
function captureBase64() {
  if (!state.stream) throw new Error("Сначала запустите камеру.");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
  // Очищаем canvas после захвата — не оставляем пиксели доступными.
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  return dataUrl.slice("data:image/jpeg;base64,".length);
}

// Возвращает Blob JPEG для multipart /liveness.
async function captureBlob() {
  if (!state.stream) throw new Error("Сначала запустите камеру.");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise((resolve) =>
    canvas.toBlob(resolve, "image/jpeg", 0.9)
  );
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  return blob;
}

// ---- Общие параметры запроса ----
function sharedParams() {
  return {
    user_id: $("#user-id").value || "demo_user_1",
    require_liveness: $("#require-liveness").checked,
    liveness_mode: $$('input[name="lmode"]:checked')[0].value,
  };
}

// ---- Health-индикаторы ----
async function refreshHealth() {
  const apiPill  = $("#health-api");
  const readyPill = $("#health-ready");
  try {
    await api("/health");
    apiPill.textContent = "API: ok";
    apiPill.className = "pill pill-ok";
  } catch {
    apiPill.textContent = "API: down";
    apiPill.className = "pill pill-bad";
  }
  try {
    const r = await api("/ready");
    if (r.status === "ok") {
      readyPill.textContent = "ready: ok";
      readyPill.className = "pill pill-ok";
    } else {
      readyPill.textContent = "ready: " + (r.status || "degraded");
      readyPill.className = "pill pill-warn";
    }
  } catch {
    readyPill.textContent = "ready: down";
    readyPill.className = "pill pill-bad";
  }
}

// ---- Отображение результатов ----
function statusBadge(status, matchScore) {
  // client-side интерпретация по слайдеру (только для match-подобных статусов)
  let clientVerdict = "";
  if (typeof matchScore === "number") {
    clientVerdict = matchScore >= state.matchThreshold ? "match" : "no_match";
  }
  const cls =
    status === "match" ? "ok"
    : status === "no_match" ? "bad"
    : status === "spoof_detected" ? "bad"
    : status === "no_face" ? "warn"
    : status === "quality_reject" ? "warn"
    : status === "retry" ? "warn"
    : status === "low_confidence" ? "warn"
    : status === "processing_failed" ? "bad"
    : "muted";
  return `<span class="badge badge-${cls}">${status || "?"}</span>` +
    (clientVerdict ? ` <span class="badge badge-muted">клиент: ${clientVerdict} (≥${state.matchThreshold.toFixed(2)})</span>` : "");
}

function bar(label, value, max = 1) {
  if (typeof value !== "number") return "";
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return `<div class="bar-row"><span class="bar-label">${label}</span>
    <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
    <span class="bar-val">${value.toFixed(3)}</span></div>`;
}

function renderVerifyResult(containerSel, data) {
  const el = $(containerSel);
  if (!data) { el.innerHTML = ""; return; }
  // Паддинг для VerifyEnqueueResponse fallback (job_id, pending)
  if (data.job_id && data.status === "pending") {
    el.innerHTML = `<p class="muted small">Поставлено в очередь. job_id=${data.job_id}. Long-poll /jobs/{id}/wait…</p>`;
    return;
  }
  const si = data.spoofing_indicators || {};
  const qd = data.quality_details || {};
  const rows = [];
  // envelope ошибки worker'а: status === "error" и result.error
  const effectiveStatus = data.status === "error" && data.result && data.result.error
    ? "processing_failed"
    : data.status;
  rows.push(`<div>${statusBadge(effectiveStatus, data.match_score ?? data.similarity)}</div>`);
  if (typeof (data.match_score ?? data.similarity) === "number")
    rows.push(bar("match_score", data.match_score ?? data.similarity));
  if (data.confidence) rows.push(`<p class="small">confidence: <b>${data.confidence}</b></p>`);
  if (typeof data.liveness_passed === "boolean")
    rows.push(`<p class="small">liveness_passed: <b>${data.liveness_passed}</b>` +
      (typeof data.liveness_score === "number" ? ` (score ${data.liveness_score.toFixed(3)})` : "") + `</p>`);
  if (data.active_note) rows.push(`<p class="small ok">${escapeHtml(data.active_note)}</p>`);
  if (typeof si.real_prob === "number") rows.push(bar("real_prob", si.real_prob));
  if (typeof si.spoof_prob === "number") rows.push(bar("spoof_prob", si.spoof_prob));
  if (data.reason) rows.push(`<p class="small">reason: ${data.reason}</p>`);
  if (data.error_code) rows.push(`<p class="small">error_code: ${data.error_code}</p>`);
  // envelope-ошибка: result.error от worker'а (152-ФЗ — без биометрии).
  if (data.result && typeof data.result.error === "string")
    rows.push(`<p class="small bad">worker error: ${escapeHtml(data.result.error)}</p>`);
  if (data.challenge_recommended)
    rows.push(`<p class="small warn">⚠ рекомендуется active challenge (серая зона)</p>`);
  if (qd && Object.keys(qd).length)
    rows.push(`<details><summary>quality_details</summary><pre class="small">${escapeHtml(JSON.stringify(qd, null, 2))}</pre></details>`);
  el.innerHTML = rows.join("");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

// ---- Verify (passive / active) ----
async function doVerify(containerSel, override = {}) {
  const el = $(containerSel);
  el.innerHTML = `<p class="muted small">обработка…</p>`;
  // active-режим: token уже пройден challenge'ом, серверный async-результат
  // liveness_passed=True не проставляет (worker не знает active_proven).
  const isActive = override.liveness_mode === "active" ||
    (!override.liveness_mode && sharedParams().liveness_mode === "active");
  try {
    const image = captureBase64();
    const body = Object.assign(
      { image },
      sharedParams(),
      override
    );
    let data = await api("/api/v1/verify_base64", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    // /verify_base64 всегда async: {job_id, status:"pending"} → long-poll /jobs/{id}/wait.
    if (data.job_id && data.status === "pending") {
      renderVerifyResult(containerSel, data);
      data = await pollJob(data.job_id);
    }
    // active: token consumed = liveness доказан; worker не знает, поэтому фиксируем клиент-side.
    if (isActive && data && typeof data === "object") {
      data.liveness_passed = true;
      data.active_note = "active challenge пройден (liveness_token consumed)";
    }
    renderVerifyResult(containerSel, data);
  } catch (e) {
    el.innerHTML = `<p class="bad small">Ошибка: ${escapeHtml(e.message)}${e.status ? ` (HTTP ${e.status})` : ""}</p>`;
  }
}

// Long-poll статуса job: GET /api/v1/jobs/{id}/wait до терминала или таймаута.
// Сервер ждёт до timeoutMs (макс 30000), после чего возвращает status=processing —
// тогда цикл повторяет wait. Общий client-side deadline — 60с (cold-start CPU).
async function pollJob(jobId, overallTimeoutMs = 60000) {
  const deadline = Date.now() + overallTimeoutMs;
  while (Date.now() < deadline) {
    let data;
    try {
      const remaining = Math.min(30000, Math.max(1000, deadline - Date.now()));
      data = await api(`/api/v1/jobs/${encodeURIComponent(jobId)}/wait?timeout=${remaining}`);
    } catch (e) {
      return { status: "processing_failed", reason: e.message };
    }
    const s = data.status;
    // Терминальные envelope-статусы worker'а: done | error | expired | failed.
    if (s === "done" || s === "error" || s === "expired" || s === "failed") {
      // Внутри envelope result лежит терминальный verify-статус (match/no_match/…).
      if (data.result && typeof data.result === "object" && data.result.status !== undefined) {
        return data.result;
      }
      // Ошибка worker'а: result.error или top-level error.
      return data;
    }
    // Не терминальный статус (processing/pending/queued/неожиданное) —
    // безусловно спим перед повтором wait. Раньше sleep был только для
    // non-pending/queued → при ответе "pending"/"queued" уходил бы в tight
    // loop без задержки (молотил бы сервер). 250мс достаточно: серверный
    // wait и так блокирует до timeout, здесь лишь страховка.
    await new Promise((r) => setTimeout(r, 250));
  }
  return { status: "processing_failed", reason: "timeout waiting for job" };
}

// ---- Upload ----
async function doUpload() {
  const el = $("#upload-result");
  el.innerHTML = `<p class="muted small">обработка…</p>`;
  try {
    const image = captureBase64();
    const data = await api("/api/v1/upload_base64", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: $("#user-id").value || "demo_user_1", image }),
    });
    const emb = data && data.data ? data.data : data;
    el.innerHTML = `<p class="ok small">Эталон записан. embedding_id=<b>${emb.embedding_id}</b>, user_id=<b>${escapeHtml(emb.user_id)}</b></p>`;
  } catch (e) {
    el.innerHTML = `<p class="bad small">Ошибка: ${escapeHtml(e.message)}${e.status ? ` (HTTP ${e.status})` : ""}</p>`;
  }
}

// Читает выбранный файл с ПК в чистый base64 (без data:-prefix) для /upload_base64.
// Возвращает { base64, contentType, name, size }. Валидирует тип/размер на клиенте
// (сервер тоже валидирует — это лишь быстрый фолбэк без круглого пути).
const MAX_UPLOAD_FILE_BYTES = 5 * 1024 * 1024; // 5 MiB — соответствует MAX_IMAGE_SIZE на сервере.

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    if (!file) return reject(new Error("Файл не выбран."));
    if (!file.type || !file.type.startsWith("image/"))
      return reject(new Error("Выберите файл-изображение (JPEG/PNG)."));
    if (file.size > MAX_UPLOAD_FILE_BYTES)
      return reject(new Error(`Файл слишком большой (${(file.size / 1024 / 1024).toFixed(1)} MiB > 5 MiB).`));
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = String(reader.result || "");
      const comma = dataUrl.indexOf(",");
      if (comma < 0) return reject(new Error("Не удалось прочитать файл."));
      resolve({
        base64: dataUrl.slice(comma + 1),
        contentType: file.type,
        name: file.name,
        size: file.size,
      });
    };
    reader.onerror = () => reject(new Error("Ошибка чтения файла."));
    reader.readAsDataURL(file);
  });
}

// Загрузка эталона из файла с ПК (вместо кадра с камеры). Тот же /upload_base64.
async function doUploadFromFile(file) {
  const el = $("#upload-result");
  el.innerHTML = `<p class="muted small">обработка файла…</p>`;
  try {
    const f = await fileToBase64(file);
    const data = await api("/api/v1/upload_base64", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: $("#user-id").value || "demo_user_1", image: f.base64 }),
    });
    const emb = data && data.data ? data.data : data;
    el.innerHTML = `<p class="ok small">Эталон записан из файла <b>${escapeHtml(f.name)}</b>. embedding_id=<b>${emb.embedding_id}</b>, user_id=<b>${escapeHtml(emb.user_id)}</b></p>`;
  } catch (e) {
    el.innerHTML = `<p class="bad small">Ошибка: ${escapeHtml(e.message)}${e.status ? ` (HTTP ${e.status})` : ""}</p>`;
  }
}

// ---- Liveness (passive, multipart) ----
async function doLiveness() {
  const el = $("#liveness-result");
  el.innerHTML = `<p class="muted small">обработка…</p>`;
  try {
    const blob = await captureBlob();
    const fd = new FormData();
    fd.append("file", blob, "frame.jpg");
    const data = await api("/api/v1/liveness", { method: "POST", body: fd });
    const si = data.spoofing_indicators || {};
    const rows = [`<span class="badge badge-${data.liveness ? "ok" : "bad"}">liveness: ${data.liveness}</span>`];
    if (typeof data.score === "number") rows.push(bar("score", data.score));
    rows.push(`<p class="small">face_detected: <b>${data.face_detected}</b></p>`);
    if (typeof si.real_prob === "number") rows.push(bar("real_prob", si.real_prob));
    if (typeof si.spoof_prob === "number") rows.push(bar("spoof_prob", si.spoof_prob));
    el.innerHTML = rows.join("");
  } catch (e) {
    el.innerHTML = `<p class="bad small">Ошибка: ${escapeHtml(e.message)}${e.status ? ` (HTTP ${e.status})` : ""}</p>`;
  }
}

// ---- Active Challenge (WS) ----
const ACTION_LABELS = {
  blink: "Моргните",
  turn_left: "Поверните голову влево",
  turn_right: "Поверните голову вправо",
  nod: "Кивните (вниз-вверх)",
  smile: "Улыбнитесь",
};

async function activeInit() {
  $("#active-result").innerHTML = "";
  $("#active-verify-result").innerHTML = "";
  state.livenessToken = null;
  $("#active-verify").disabled = true;
  try {
    const data = await api("/api/v1/liveness/challenge/init", { method: "POST" });
    renderActions(data.actions || []);
    $("#active-status").textContent = "Challenge инициализирован. Выполните действия — кадры стримятся автоматически.";
    openStream(data);
  } catch (e) {
    $("#active-status").textContent = "Ошибка init: " + e.message +
      (e.status === 503 ? " (проверьте LIVENESS_ENABLED / LIVENESS_ACTIVE_ENABLED)" : "");
    $("#active-actions").innerHTML = "";
  }
}

function renderActions(actions) {
  $("#active-actions").innerHTML = actions
    .map((a) => `<div class="action-item">→ ${ACTION_LABELS[a] || a}</div>`)
    .join("");
}

function openStream(initData) {
  const wsUrl = (location.protocol === "https:" ? "wss:" : "ws:") + "//" + location.host + initData.ws_url;
  state.ws = new WebSocket(wsUrl);
  state.ws.binaryType = "arraybuffer";
  state.wsFramesSent = 0;

  state.ws.onopen = () => {
    // «Готово» включается только после MIN_FRAMES_BEFORE_DONE отправленных
    // кадров — иначе вердикт падает в too_few_frames (серверу нужно ≥6 кадров
    // С лицом; 600мс-интервал + ранний «Готово» давали всего ~5).
    $("#active-done").disabled = true;
    $("#active-cancel").disabled = false;
    $("#active-init").disabled = true;
    // Стримим кадры каждые ~300мс (быстрее ловятся быстрые действия — моргание;
    // лимит 30 кадров / 60с на сервере).
    state.wsInterval = setInterval(streamFrame, 300);
    updateStreamStatus();
  };

  state.ws.onmessage = async (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "challenge") {
      $("#active-status").textContent = `Стрим активен. Действия: ${(msg.actions || []).join(", ")}. deadline ${msg.deadline_ms}мс`;
    } else if (msg.type === "result") {
      handleResult(msg);
    }
  };

  state.ws.onclose = (ev) => {
    cleanupStream();
    if (!state.livenessToken) {
      const reason = wsCloseReason(ev.code);
      if (reason) $("#active-status").textContent = "Соединение закрыто: " + reason;
    }
  };

  state.ws.onerror = () => {
    $("#active-status").textContent = "Ошибка WebSocket.";
  };
}

async function streamFrame() {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  if (state.wsFramesSent >= 30) { // LIVENESS_WS_MAX_FRAMES
    stopStreaming();
    return;
  }
  try {
    const blob = await captureBlob();
    const buf = await blob.arrayBuffer();
    state.ws.send(buf);
    state.wsFramesSent++;
    updateStreamStatus();
  } catch (e) {
    // камера не готова — пропускаем кадр
  }
}

// Сколько кадров клиент должен отправить прежде, чем даст нажать «Готово».
// Серверу нужно ≥6 кадров С лицом (LIVENESS_MIN_FRAMES); берём с запасом,
// т.к. часть кадров может прийти без лица (observe_frame → None → пропуск).
const MIN_FRAMES_BEFORE_DONE = 10;

function updateStreamStatus() {
  const sent = state.wsFramesSent;
  if (sent < MIN_FRAMES_BEFORE_DONE) {
    $("#active-done").disabled = true;
    $("#active-status").textContent =
      `Стрим: отправлено ${sent}/${MIN_FRAMES_BEFORE_DONE} кадров. Держите лицо в кадре и медленно выполните действия — «Готово» станет активным.`;
  } else {
    $("#active-done").disabled = false;
    $("#active-status").textContent =
      `Стрим: отправлено ${sent} кадров. Выполните все показанные действия и нажмите «Готово».`;
  }
}

function stopStreaming() {
  if (state.wsInterval) { clearInterval(state.wsInterval); state.wsInterval = null; }
}

function cleanupStream() {
  stopStreaming();
  state.ws = null;
  $("#active-done").disabled = true;
  $("#active-cancel").disabled = true;
  $("#active-init").disabled = false;
}

function handleResult(msg) {
  stopStreaming();
  const live = msg.is_live;
  state.livenessToken = msg.liveness_token || null;
  const si = msg.spoofing_indicators || {};
  const reason = msg.reason || si.reason || "";
  const nFrames = typeof msg.n_frames === "number" ? msg.n_frames : null;
  const rows = [`<span class="badge badge-${live ? "ok" : "bad"}">is_live: ${live}</span>`];
  if (state.livenessToken) {
    rows.push(`<p class="ok small">liveness_token получен (в памяти, TTL 120с). Можно «Verify active».</p>`);
  } else {
    rows.push(`<p class="bad small">liveness_token не выдан. ${reason ? "Причина: " + reason : ""}</p>`);
  }
  // too_few_frames — consistency не вычислялась (early-return на сервере),
  // показывать «consistency: fail» тут вводит в заблуждение — поясняем кадрами.
  if (reason === "too_few_frames") {
    rows.push(`<p class="small">Слишком мало кадров с лицом${nFrames !== null ? ` (${nFrames}/6)` : ""}. Держите лицо в кадре и стримите дольше до «Готово».</p>`);
  } else if (si.consistency) {
    rows.push(`<p class="small">consistency: ${si.consistency}</p>`);
  }
  $("#active-result").innerHTML = rows.join("");
  $("#active-verify").disabled = !state.livenessToken;
  $("#active-status").textContent = live ? "Challenge пройден." : "Challenge не пройден — спуфинг или действия не выполнены.";
  try { state.ws.close(); } catch {}
  cleanupStream();
}

function activeDone() {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  if (state.wsFramesSent < MIN_FRAMES_BEFORE_DONE) {
    $("#active-status").textContent =
      `Слишком мало кадров (${state.wsFramesSent}/${MIN_FRAMES_BEFORE_DONE}). Подождите, пока набор достаточно, затем «Готово».`;
    return;
  }
  stopStreaming();
  state.ws.send(JSON.stringify({ cmd: "done" }));
  $("#active-status").textContent = "Отправлен cmd:done — ожидание вердикта…";
}

function activeCancel() {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ cmd: "cancel" }));
  }
  try { state.ws && state.ws.close(); } catch {}
  cleanupStream();
  $("#active-status").textContent = "Challenge отменён.";
}

function activeVerify() {
  if (!state.livenessToken) return;
  doVerify("#active-verify-result", {
    require_liveness: true,
    liveness_mode: "active",
    liveness_token: state.livenessToken,
  });
}

function wsCloseReason(code) {
  switch (code) {
    case 4410: return "challenge истёк или уже использован — начните заново";
    case 4401: return "неверный ws_token — начните заново";
    case 4503: return "сервер занят или active liveness отключён";
    case 4409: return "challenge уже стримится (конфликт)";
    case 1006: return "соединение разорвано";
    default: return code ? `код ${code}` : "";
  }
}

// ---- Config ----
async function refreshConfig() {
  const tbody = $("#config-table tbody");
  try {
    const cfg = await api("/api/v1/config");
    state.config = cfg;
    const rows = [
      ["FACE_MATCH_THRESHOLD", cfg.FACE_MATCH_THRESHOLD],
      ["LIVENESS_THRESHOLD", cfg.LIVENESS_THRESHOLD],
      ["LIVENESS_ENABLED", cfg.LIVENESS_ENABLED],
      ["LIVENESS_ACTIVE_ENABLED", cfg.LIVENESS_ACTIVE_ENABLED],
      ["LIVENESS_ACTIVE_REQUIRED", cfg.LIVENESS_ACTIVE_REQUIRED],
      ["QUALITY_GATE_MODE", cfg.QUALITY_GATE_MODE],
    ];
    tbody.innerHTML = rows.map(([k, v]) =>
      `<tr><td>${k}</td><td>${escapeHtml(String(v))}</td></tr>`).join("");
    // Слайдер по умолчанию = серверному порогу
    if (typeof cfg.FACE_MATCH_THRESHOLD === "number") {
      $("#match-slider").value = cfg.FACE_MATCH_THRESHOLD;
      $("#match-slider-val").textContent = cfg.FACE_MATCH_THRESHOLD.toFixed(2);
      state.matchThreshold = cfg.FACE_MATCH_THRESHOLD;
    }
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="2" class="bad">Ошибка: ${escapeHtml(e.message)}</td></tr>`;
  }
}

// ---- Табы ----
function initTabs() {
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".tab").forEach((t) => t.classList.remove("active"));
      $$(".tab-panel").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      $(`.tab-panel[data-panel="${tab.dataset.tab}"]`).classList.add("active");
    });
  });
}

// ---- init ----
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  refreshHealth();
  refreshConfig();
  setInterval(refreshHealth, 10000);

  $("#cam-start").addEventListener("click", camStart);
  $("#cam-stop").addEventListener("click", camStop);

  $("#verify-run").addEventListener("click", () => doVerify("#verify-result"));
  $("#upload-run").addEventListener("click", doUpload);
  $("#upload-file").addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    const nameEl = $("#upload-file-name");
    if (file) {
      if (nameEl) nameEl.textContent = `Выбран: ${file.name} (${(file.size / 1024).toFixed(0)} КБ)`;
      doUploadFromFile(file);
    } else if (nameEl) {
      nameEl.textContent = "";
    }
    // Сбрасываем input, чтобы тот же файл можно было выбрать повторно.
    e.target.value = "";
  });
  $("#liveness-run").addEventListener("click", doLiveness);

  $("#active-init").addEventListener("click", activeInit);
  $("#active-done").addEventListener("click", activeDone);
  $("#active-cancel").addEventListener("click", activeCancel);
  $("#active-verify").addEventListener("click", activeVerify);

  $("#config-refresh").addEventListener("click", refreshConfig);
  $("#match-slider").addEventListener("input", (e) => {
    state.matchThreshold = parseFloat(e.target.value);
    $("#match-slider-val").textContent = state.matchThreshold.toFixed(2);
  });

  // При выборе active-режима в общих параметрах — направляем в active-таб
  $$('input[name="lmode"]').forEach((r) => {
    r.addEventListener("change", (e) => {
      if (e.target.value === "active") {
        $$('.tab[data-tab="active"]')[0].click();
      }
    });
  });
});

window.addEventListener("beforeunload", () => {
  camStop();
  if (state.ws) { try { state.ws.close(); } catch {} }
});