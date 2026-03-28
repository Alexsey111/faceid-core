# FaceID Core Dashboard Guide

## How To Read The Dashboard

Benchmark note:
- The first valid E2E sweep was used to validate the queue policy and `/wait` shape.
- The next scaling pass should read `queue_delay`, `processing_time`, `total_latency`, and drain rate from Prometheus instead of treating `/wait` as the primary measurement source.
- That makes the comparison less sensitive to polling behavior and client-side wait time.

### 1. Queue Delay Is The Main Signal
- If `queue_delay_p95` goes up, the workers are overloaded.
- This is the primary CPU saturation signal.
- If `queue_delay_p95` rises while `pipeline_p95` stays flat, the problem is the queue, not ML.

### 2. Latency Versus Pipeline
- If `pipeline_p95` is stable but total latency keeps rising, requests are waiting before processing starts.
- That usually means queue pressure or not enough worker capacity.
- If `pipeline_p95` rises, the slowdown is inside the ML pipeline itself.

### 3. Quality Reject Rate
- If `quality_reject_rate` rises sharply, one of two things is happening:
- The thresholds are too strict.
- The traffic quality got worse: blur, dark images, tiny faces, or junk inputs.
- Always check reject reasons first.

### 4. Detect Versus Encode
- If `encode_p95` is much higher than `detect_p95`, that is usually normal.
- ArcFace embedding extraction is expensive.
- If `detect_p95` starts growing, the detector path is the problem.

### 5. Quality Gate Timing
- `quality_gate_pre_ms` and `quality_gate_face_ms` should stay fast.
- If either one goes above `30 ms`, the gate starts eating CPU.
- A healthy target is around `5-15 ms`.

## How To Make Decisions

### Scenario 1: System Is Slow
Signals:
- `queue_delay_p95` rises
- `pipeline_p95` stays stable

Action:
- Scale workers
- Scale horizontally

### Scenario 2: Too Many Rejects
Signals:
- `quality_reject_rate` rises

Action:
- Relax thresholds
- Start with:
- `MIN_BLUR_SCORE`
- `MIN_BRIGHTNESS`
- `MIN_FACE_SIDE`

### Scenario 3: Latency Is Increasing
Signals:
- `pipeline_p95` rises

Action:
- The problem is in ML, not the queue
- Check detector, preprocessing, encode, and quality gate timing

## Minimum SLO
- `p95 total_latency < 2s`
- `queue_delay_p95 < 1s`
- `quality_reject < 20%`

## Short Version
- Queue delay up means capacity is the bottleneck.
- Pipeline latency up means ML is the bottleneck.
- Quality rejects up means thresholds or traffic quality changed.
- Detect latency up means the detector path is under stress.
- Encode latency up is expected to be the expensive part, but it still needs watching.
