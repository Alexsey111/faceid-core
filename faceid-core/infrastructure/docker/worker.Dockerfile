# D:\python projects\faceid-core\faceid-core\infrastructure\docker\worker.Dockerfile
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

COPY faceid-core/app ./app

CMD ["celery", "-A", "app.workers.celery_app", "worker", "--loglevel=info", "--concurrency=6", "--prefetch-multiplier=1"]
