# FaceID Core

## SLA And Load Profile

The system has two different verification paths with different service goals:

- `/verify`
  - synchronous path
  - target: low latency
  - should be treated as a strict user-facing endpoint
- `/verify_async`
  - queued path
  - target: throughput and backpressure control
  - latency is allowed to degrade under load

### FaceID Core v1.0 baseline

This release should be treated as a stable, controlled baseline:

- `RATE <= 3` -> OK
- `RATE ~= 5` -> degraded, but controlled
- `RATE >= 6` -> stress, `429` responses are expected

Current operating limits:

- CPU-only inference
- adaptive batching enabled
- backpressure enabled with `750 ms` queue-delay guard

Important:

- `429` is not an application error here
- it is the protection mechanism that keeps queue delay from growing without bound
- this behavior is part of the design, not a temporary workaround

### Passive liveness artifact

Canonical model file:

- `models/liveness.onnx`

Legacy-compatible fallback:

- `models/antispoof.onnx`

Runtime behavior:

- Prefer `liveness.onnx`
- Fall back to `antispoof.onnx`
- Disable liveness gracefully if neither file exists

Recommended deployment flow:

1. Copy the ONNX model into `models/liveness.onnx`
2. Rebuild and restart the stack
3. Verify the worker metrics endpoint exposes `faceid_liveness_ms_*`

### Operational load bands

The async path is intentionally tuned around the following load bands:

| `RATE` | Mode | Expected behavior |
|---|---|---|
| `RATE <= 2` | Stable low latency | Queue delay should stay short and the system should feel responsive |
| `RATE ~= 5` | Working mode | This is the intended steady-state operating point |
| `RATE >= 8` | Stress | Degradation is acceptable, but the system should still reject early instead of building an unbounded queue |

### Backpressure rules

- Reject when estimated queue delay would exceed `750 ms`
- Keep batching adaptive:
  - bypass batching at low load
  - batch at moderate/high load
  - flush a batch early if the oldest request waits too long
- Keep test and production databases isolated

### Test setup

Integration and end-to-end tests run against a separate PostgreSQL instance on port `5433` and apply migrations with Alembic before the suite starts.

### Local verification

```bash
pytest -q
```
