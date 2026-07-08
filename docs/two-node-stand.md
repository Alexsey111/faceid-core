# Two-node stand

Node 1 runs API, Redis, Postgres, MinIO, and the local worker pool.
Node 2 runs only the worker pool and connects back to node 1 over the network.

## Node 1

Use the existing `docker-compose.yml`.

```bash
docker compose up -d
```

If you want to pin the async worker count explicitly:

```bash
docker compose up -d --scale worker=4
```

If node 1 has an NVIDIA GPU, layer the GPU override (CUDA build + GPU reservations
+ `ONNX_ARCFACE_PROVIDERS=cuda`); see `docs/deploy-runbook.md` § GPU-сборка:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

## Node 2

1. Create a node 2 env file from `docker-compose.node2.env.example`.
1. Replace the example IPs with the real node 1 address.
1. Start the worker stack.

```bash
docker compose -f docker-compose.node2.yml --env-file docker-compose.node2.env up -d --scale worker=4
```

## Notes

- Node 2 does not run Redis, Postgres, or MinIO locally.
- The worker container needs network access to node 1 ports `5432`, `6379`, and `9000`.
