# worker.Dockerfile — verify_worker (CPU, async-native). См. docs/deploy-runbook.md.
# Celery удалён (Волна C): единый production-путь — app.workers.verify_worker
# (Redis-очередь, batch, backpressure). Dockerfile default-CMD совпадает с
# compose `command:` (compose переопределял CMD, но образ должен стартовать сам).
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models

CMD ["python", "-m", "app.workers.verify_worker"]
