# api.gpu.Dockerfile — API-образ для CUDA-хостов (production с GPU).
# База: nvidia/cuda 11.8 + cuDNN 8 (требование onnxruntime-gpu==1.18.0).
# Python 3.11 ставится из deadsnakes PPA (в CUDA-образе его нет).
# См. docs/deploy-runbook.md — раздел «GPU-сборка».
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Системные зависимости + Python 3.11 (deadsnakes) + pip.
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

WORKDIR /app

# Копирование и установка зависимостей (GPU-вариант).
COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

# Копирование кода приложения
COPY app ./app
COPY models ./models

# Команда запуска (переопределяется в docker-compose.yml)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "3"]