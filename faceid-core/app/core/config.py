# app/core/config.py

import base64
import math

from functools import lru_cache
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    ENV: str = "development"
    APP_ROLE: str = "api"

    APP_NAME: str = "FaceID Core"
    DEBUG: bool = False

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() not in ("false", "0", "no", "release")
        return bool(v)

    # -------------------------
    # PostgreSQL
    # -------------------------
    POSTGRES_HOST: str = "postgres"
    DB_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "faceid"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"

    # -------------------------
    # Redis
    # -------------------------
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_ENABLED: bool = True

    # -------------------------
    # MinIO
    # -------------------------
    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "face-images"
    MINIO_SECURE: bool = False

    # -------------------------
    # Models
    # -------------------------
    MODELS_DIR: str = "D:/python projects/faceid-core"
    ONNX_INTRA_OP_THREADS: int = 2
    ONNX_INTER_OP_THREADS: int = 1

    # -------------------------
    # Search backend
    # -------------------------
    SEARCH_BACKEND: str = "pgvector"  # pgvector | faiss

    # -------------------------
    # FAISS
    # -------------------------
    FAISS_ENABLED: bool = True
    FAISS_INDEX_PATH: str = "faiss.index"
    FAISS_PERSIST_ENABLED: bool = True

    # -------------------------
    # Face verification thresholds
    # -------------------------
    PREPROCESS_MAX_SIDE: int = 640
    RETINA_DET_SIZE: int = 320
    USE_PIPELINE_V2: bool = True
    USE_SIMPLE_IS_GENUINE: bool = True
    USE_FAST_PATH: bool = True
    FAST_PATH_MAX_CONCURRENCY: int = 4
    FAST_WORKER_URL: str = "http://fast_worker:8000"
    FAST_WORKER_ENABLED: bool = True
    FAST_WORKER_FAILURES: int = 0
    FAST_WORKER_MAX_FAILURES: int = 3
    FAST_WORKER_MAX_CONCURRENCY: int = 4
    LIVENESS_ENABLED: bool = False
    LIVENESS_THRESHOLD: float = 0.5
    FACE_MATCH_THRESHOLD: float = 0.6
    FACE_LOW_THRESHOLD: float = 0.3
    FACE_MARGIN_THRESHOLD: float = 0.1
    HIGH_THRESHOLD: float = 0.60
    LOW_THRESHOLD: float = 0.30
    MARGIN_THRESHOLD: float = 0.10

    # -------------------------
    # Security
    # -------------------------
    SECRET_KEY: str = "fgLuHbGN6wiyxT-pfDqe6QBsP8nsf-KpZ3IzV-wCnn4="
    AES_SECRET_KEY: str = "0123456789abcdef0123456789abcdef"
    BIOMETRY_AES_KEY_B64: str | None = None

    # -------------------------
    # Celery
    # -------------------------
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"
    CELERY_TASK_QUEUE: str = "faceid"
    WORKER_COUNT: int = 3
    ACTIVE_TASKS_MULTIPLIER: float = 1.25
    ASYNC_THROUGHPUT_PER_SEC: float = 8.0
    BACKPRESSURE_MAX_QUEUE_DELAY_MS: float = 10000.0
    WORKER_SEMAPHORE: int = 2
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 1800
    EMBED_BATCH_ENABLED: bool = True
    EMBED_BATCH_SIZE: int = 8
    EMBED_BATCH_TIMEOUT_MS: float = 2.0
    EMBED_BATCH_MAX_WAIT_GUARD_MS: float = 50.0

    @property
    def MAX_ACTIVE_TASKS(self) -> int:
        return max(1, math.ceil(self.WORKER_COUNT * self.ACTIVE_TASKS_MULTIPLIER))

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # -------------------------
    # Computed properties
    # -------------------------

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://"
            f"{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def async_database_url(self) -> str:
        return (
            f"postgresql+asyncpg://"
            f"{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def DATABASE_URL(self) -> str:
        return self.database_url

    @property
    def redis_url(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def is_production(self) -> bool:
        return self.ENV == "production"

    @property
    def aes_key(self) -> bytes:
        key = self.BIOMETRY_AES_KEY_B64
        if self.is_production and not key:
            raise RuntimeError("AES key required in production")
        if not key:
            return b"0" * 32
        return base64.b64decode(key)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
