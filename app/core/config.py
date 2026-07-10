# app/core/config.py

import base64
import logging
import math

from functools import lru_cache
from pydantic import AliasChoices, BaseModel, Field
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("config")


class AdmissionSettingsSnapshot(BaseModel):
    inflight_limit: int | None
    max_queue_size: int | None
    backpressure_max_queue_delay_ms: int | None
    async_throughput_per_sec: float | None
    max_job_age_ms: int | None
    max_queue_wait: int | None
    worker_count: int | None
    worker_semaphore: int | None
    embed_batch_size: int | None
    embed_batch_timeout_ms: int | None
    api_uvicorn_workers: int | None


class ServiceRuntimeSnapshot(BaseModel):
    service: str
    build_version: str | None
    git_sha: str | None
    image_tag: str | None
    admission_settings: AdmissionSettingsSnapshot
    uvicorn_workers: int | None
    worker_replicas: int | None
    worker_count: int | None
    worker_semaphore: int | None
    worker_batch_size: int | None
    worker_batch_collect_timeout_ms: int | None
    embed_batch_size: int | None
    embed_batch_timeout_ms: int | None


class Settings(BaseSettings):

    ENV: str = "development"
    APP_ROLE: str = "api"

    APP_NAME: str = "FaceID Core"
    DEBUG: bool = False
    APP_VERSION: str | None = Field(
        default=None,
        validation_alias=AliasChoices("APP_VERSION", "VERSION"),
    )
    GIT_SHA: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GIT_SHA", "GITHUB_SHA", "COMMIT_SHA"),
    )
    IMAGE_TAG: str | None = Field(
        default=None,
        validation_alias=AliasChoices("IMAGE_TAG", "DOCKER_IMAGE_TAG"),
    )

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() not in ("false", "0", "no", "release")
        return bool(v)

    @field_validator("QUALITY_GATE_MODE", mode="before")
    @classmethod
    def parse_quality_gate_mode(cls, v):
        if v is None:
            return "hard"
        value = str(v).strip().lower()
        if value not in {"hard", "soft", "off"}:
            raise ValueError("QUALITY_GATE_MODE must be one of: hard, soft, off")
        return value

    @field_validator("POSE_QUALITY_MODE", mode="before")
    @classmethod
    def parse_pose_quality_mode(cls, v):
        if v is None:
            return "soft"
        value = str(v).strip().lower()
        if value not in {"hard", "soft", "off"}:
            raise ValueError("POSE_QUALITY_MODE must be one of: hard, soft, off")
        return value

    @field_validator("QUALITY_LIGHTING_MODE", mode="before")
    @classmethod
    def parse_quality_lighting_mode(cls, v):
        if v is None:
            return "soft"
        value = str(v).strip().lower()
        if value not in {"hard", "soft", "off"}:
            raise ValueError("QUALITY_LIGHTING_MODE must be one of: hard, soft, off")
        return value

    @field_validator("QUALITY_NOISE_MODE", mode="before")
    @classmethod
    def parse_quality_noise_mode(cls, v):
        if v is None:
            return "off"
        value = str(v).strip().lower()
        if value not in {"hard", "soft", "off"}:
            raise ValueError("QUALITY_NOISE_MODE must be one of: hard, soft, off")
        return value

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
    # Провайдеры ArcFace-энкодера. Варианты:
    #   "auto"      → CUDA → DirectML → CPU (по доступности, с fallback)
    #   "cuda"      → ["CUDAExecutionProvider","CPUExecutionProvider"]
    #   "directml"  → ["DmlExecutionProvider","CPUExecutionProvider"]
    #   "cpu"       → ["CPUExecutionProvider"]
    #   явный csv   → напр. "CUDAExecutionProvider,CPUExecutionProvider"
    # На машине без GPU (только CPUExecutionProvider) честно падает в CPU.
    ONNX_ARCFACE_PROVIDERS: str = "auto"
    # Путь к ArcFace recognition-модели относительно MODELS_DIR
    # (детектор buffalo_l/scrfd — отдельный pack, им пользуется FaceAnalysis).
    ARCFACE_MODEL_REL: str = "buffalo_l/w600k_r50.onnx"

    # -------------------------
    # Search backend
    # -------------------------
    SEARCH_BACKEND: str = "pgvector"  # pgvector | faiss

    # -------------------------
    # FAISS
    # -------------------------
    FAISS_ENABLED: bool = True
    FAISS_INDEX_PATH: str = "faiss.index"
    # Выключено по умолчанию: persist-файл faiss.index хранит plaintext-биометрию.
    # Индекс перестраивается из encrypted_embedding при старте (см. faiss_loader).
    FAISS_PERSIST_ENABLED: bool = False

    # -------------------------
    # Face verification thresholds
    # -------------------------
    PREPROCESS_MAX_SIDE: int = 480
    # -------------------------
    # Quality gate
    # -------------------------
    QUALITY_GATE_MODE: str = "hard"  # hard | soft | off
    QUALITY_MIN_IMAGE_SIDE: int = 160
    QUALITY_MIN_BLUR_SCORE: float = 45.0
    QUALITY_MIN_BRIGHTNESS: float = 35.0
    QUALITY_MAX_BRIGHTNESS: float = 225.0
    QUALITY_MIN_CONTRAST: float = 18.0
    QUALITY_MIN_FACE_SIDE: int = 72
    QUALITY_MAX_EYE_LINE_DIFF_RATIO: float = 0.12
    QUALITY_MAX_NOSE_OFFSET_RATIO: float = 0.18
    POSE_QUALITY_MODE: str = "soft"  # hard | soft | off — отдельный режим pose-check
    # Lighting (capture-качество): равномерность освещения и жёсткая тень. Свой режим,
    # независимый от QUALITY_GATE_MODE (как pose). soft → warning-only (не отбрасывает,
    # бережёт TAR на боковом свете), hard → quality_reject, off → пропустить.
    QUALITY_LIGHTING_MODE: str = "soft"  # hard | soft | off
    QUALITY_MIN_LIGHTING_UNIFORMITY: float = 0.55  # min_cell/max_cell по сетке 3×3
    QUALITY_MAX_SHADOW_ASYMMETRY: float = 0.30  # |mean_left-mean_right|/overall_mean
    # Шум (capture-качество): ISO-noise не ловится blur-gate (Laplacian variance на
    # шумном фото ложноположительно высок → проходит blur). Метрика = std residual
    # после medianBlur(3) (классический noise-estimator: high-freq стохастика).
    # Свой режим, default off (бережёт TAR на бюджетных камерах; hard → quality_reject).
    QUALITY_NOISE_MODE: str = "off"  # hard | soft | off
    QUALITY_MAX_NOISE_STD: float = 12.0  # std residual gray-medianBlur(3), шкала 0-255
    # Окклюзия (маска/очки) — НЕ capture-качество, а требование чистого лица для
    # допуска. Детекция всегда; при срабатывании → status="retry", reason=
    # "remove_occlusion" (просим снять и пере-снять). Режим hard/soft/off НЕ действует:
    # retry всегда. Тумблеры позволяют отключить детекцию отдельно.
    QUALITY_MASK_DETECT_ENABLED: bool = True
    QUALITY_MIN_LOWER_FACE_SKIN_FRAC: float = 0.45  # ниже → mask_detected
    QUALITY_GLASSES_DETECT_ENABLED: bool = True
    QUALITY_MAX_EYE_EDGE_DENSITY: float = 0.25  # Sobel-magnitude mean/255; выше → glasses_detected
    # Солнцезащитные очки (тёмные/среднепрозрачные линзы): edge-density по оправе
    # их не ловит (гладкая тёмная линза без сильных краёв) → кадр уходит в passive
    # liveness, где MiniFASNet клеймит затенённые глаза как spoof (ложный reject
    # легального пользователя). Сигнал: глазная зона темнее подглазной/скуловой
    # (ratio eye_band_mean / cheek_band_mean). Ниже порога → sunglasses_detected →
    # retry/remove_occlusion (как маска: «снимите очки»), а не spoof.
    QUALITY_DARK_EYES_DETECT_ENABLED: bool = True
    QUALITY_MAX_EYE_DARK_RATIO: float = 0.77  # eye/cheek brightness; ниже → sunglasses_detected
    # Калибровано на веб-камере c110 (5 в очках / 5 без): очки 0.587–0.702,
    # без очков 0.842–1.097, зазор 0.140. 0.77 — центр зазора (запасы ~0.07 с
    # обеих сторон). На других камерах/лицах диапазон может плавать — при росте
    # False Retry/Pass пересобрать калибровку на 20–30 лицах.
    # «Серая» зона margin верификации: при попадании в неё ok-ответ несёт
    # challenge_recommended=True — клиент зовёт WS active-challenge (turn/nod).
    CHALLENGE_MARGIN_LOW: float = 0.05
    CHALLENGE_MARGIN_HIGH: float = 0.20
    RETINA_DET_SIZE: int = 512
    RETINA_DET_SIZE_SMALL: int = 320
    USE_SIMPLE_IS_GENUINE: bool = True
    # Лимит concurrent ML-инференса в API-процессе (роут /verify). Раньше был
    # связан с FAST_WORKER_MAX_CONCURRENCY (sync fast-path), но fast_worker
    # удалён — путь верификации единый async (face_verify_queue → worker).
    API_INFER_CONCURRENCY: int = 4
    LIVENESS_ENABLED: bool = False
    # Порог решения liveness (применяет caller, не чекер). 0.859 — рекомендация
    # eval-harness (argmin ACER на Anti-Spoofing Dataset): при нём APCER_max≈0.21,
    # NPCER≈0.086. Для «пропустить максимум живых» можно снизить до 0.7–0.8.
    LIVENESS_THRESHOLD: float = 0.859
    # Контракт yakhyo MiniFASNetV2: квадратный кроп scale×стороны bbox, вход 80×80.
    LIVENESS_CROP_SCALE: float = 2.7
    LIVENESS_INPUT_SIZE: int = 80
    # -------------------------
    # Active challenge liveness (online access control)
    # -------------------------
    # Feature-flag активного challenge-протокола (WS-стрим). Passive /liveness
    # работает независимо от этого флага.
    LIVENESS_ACTIVE_ENABLED: bool = False
    # Обязательный active-challenge gate допуска: если True — /verify с
    # require_liveness=true принимает ТОЛЬКО active proof (liveness_token из
    # /liveness/challenge/stream); passive-запрос на допуск → 403
    # active_liveness_required. Закрывает physical-spoof (cutout/print/replay),
    # который ложит passive-модель MiniFASNetV2 (cutout→real P=0.976). Default
    # False (backward-compat); high-security deploys ставят true в env
    # (требует также LIVENESS_ACTIVE_ENABLED=true).
    LIVENESS_ACTIVE_REQUIRED: bool = False
    LIVENESS_CHALLENGE_TTL_S: int = 60     # окно на запись/стрим видео после /init
    LIVENESS_TOKEN_TTL_S: int = 120        # окно действия liveness_token в /verify
    LIVENESS_CHALLENGE_ACTIONS: int = 2    # сколько действий в challenge
    LIVENESS_WS_MAX_CONCURRENT: int = 8    # бёрст-лимит параллельных WS-сессий
    LIVENESS_WS_MAX_FRAMES: int = 30       # лимит кадров на сессию (≤3с @10fps)
    # 106-pt 2D landmarks (2d106det, часть pack'а buffalo_l) для EAR-blink и pose.
    LIVENESS_LANDMARK_MODEL_REL: str = "buffalo_l/2d106det.onnx"
    # Пороги детекции действий (эмпирические, калибруются на scaling-этапе):
    LIVENESS_YAW_MIN_DEG: float = 25.0          # экскурсия yaw для turn_left/right
    LIVENESS_PITCH_MIN_EXCURSION: float = 0.10  # отклонение nose_rel для nod
    LIVENESS_EAR_DIP_RATIO: float = 0.30        # EAR падает на ≥30% от baseline → blink
    LIVENESS_SMILE_DELTA: float = 0.04          # прирост mouth_width_ratio → smile
    LIVENESS_CONSISTENCY_AREA_CV: float = 0.25  # max CV площади bbox по последовательности
    LIVENESS_CONSISTENCY_IOU_MIN: float = 0.30  # min IoU bbox кадр-к-кадру (анти jump-cut)
    LIVENESS_MIN_FRAMES: int = 6                # минимум кадров с лицом для вердикта
    # Порог допуска match. Калиброван под ТЗ FRR≤3% на LFW single-face
    # (docs/recognition-accuracy-assessment.md): при 0.45 FRR=2.71%, FAR=0.
    # Прежнее 0.6 давало FRR=22% (безопасно, но каждый 5-й legit отказан).
    # LOW_THRESHOLD=0.3 — no_match; [0.3,0.45) — low_confidence (challenge_recommended).
    FACE_MATCH_THRESHOLD: float = 0.45
    FACE_LOW_THRESHOLD: float = 0.3
    FACE_MARGIN_THRESHOLD: float = 0.1
    HIGH_THRESHOLD: float = 0.45
    LOW_THRESHOLD: float = 0.30
    MARGIN_THRESHOLD: float = 0.10
    # Pre-filter поиска в SearchService: кандидаты ниже этого порога отбрасываются
    # до top-k/decision. Выровнен с LOW_THRESHOLD (no_match), а НЕ с HIGH_THRESHOLD:
    # pre-filter должен срезать только явные no_match (<0.3), оставляя весь
    # low_confidence-диапазон (0.3–0.6) достижимым для make_decision. Раньше был
    # скрытый fallback getattr(settings, "SIM_THRESHOLD", 0.5) — «съедал» 0.3–0.5.
    SIM_THRESHOLD: float = 0.30

    # -------------------------
    # Security
    # -------------------------
    SECRET_KEY: str = "fgLuHbGN6wiyxT-pfDqe6QBsP8nsf-KpZ3IzV-wCnn4="
    AES_SECRET_KEY: str = "0123456789abcdef0123456789abcdef"
    BIOMETRY_AES_KEY_B64: str | None = None

    # -------------------------
    # Auth (ТЗ 3.1: защита эндпоинтов)
    # -------------------------
    # Мастер-переключатель: False → require_auth коротко замыкается (testing/dev).
    AUTH_ENABLED: bool = True
    JWT_ALG: str = "HS256"
    JWT_ISSUER: str | None = None
    JWT_AUDIENCE: str | None = None
    # Секрет HS256. Если не задан — fallback на SECRET_KEY (с warning в production).
    JWT_SECRET: str | None = None
    # Service-to-service ключи, CSV: "key1,key2" → множество для O(1)-проверки.
    API_KEYS: str = ""

    # -------------------------
    # Webhook (ТЗ 3.2: уведомления внешних систем о завершении верификации)
    # -------------------------
    # Мастер-переключатель: False → WebhookService.notify — no-op.
    WEBHOOK_ENABLED: bool = False
    # URL приёмника (https://...). Обязателен, если WEBHOOK_ENABLED=True.
    WEBHOOK_URL: str | None = None
    # Секрет для HMAC SHA-256 подписи тела (заголовок X-FaceID-Signature).
    WEBHOOK_SECRET: str | None = None
    # Таймаут одной HTTP-попытки, секунды.
    WEBHOOK_TIMEOUT_S: float = 5.0
    # Число попыток доставки (экспоненциальный backoff 2^attempt).
    WEBHOOK_MAX_RETRIES: int = 3
    # Размер in-process очереди доставок (fire-and-forget, bounded).
    WEBHOOK_QUEUE_SIZE: int = 256
    # TTL Redis-ключа идемпотентности webhook:sent:{job_id} (секунды).
    WEBHOOK_IDEMPOTENCY_TTL_S: int = 3600

    # -------------------------
    # Celery
    # -------------------------
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"
    CELERY_TASK_QUEUE: str = "faceid"
    WORKER_COUNT: int = 3
    ACTIVE_TASKS_MULTIPLIER: float = 1.25
    ASYNC_THROUGHPUT_PER_SEC: float = 1000.0
    BACKPRESSURE_MAX_QUEUE_DELAY_MS: float = 10000.0
    WORKER_SEMAPHORE: int = 2
    INFLIGHT_LIMIT: int | None = None
    MAX_QUEUE_SIZE: int | None = None
    MAX_JOB_AGE_MS: int | None = None
    ADMISSION_ACTIVE_SERVICE_SLOTS: int | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "ADMISSION_ACTIVE_SERVICE_SLOTS",
            "ACTIVE_SERVICE_SLOTS",
        ),
    )
    MAX_QUEUE_WAIT: int | None = Field(
        default=None,
        validation_alias=AliasChoices("MAX_QUEUE_WAIT", "MAX_QUEUE_WAIT_SEC"),
    )
    API_UVICORN_WORKERS: int | None = Field(
        default=None,
        validation_alias=AliasChoices("API_UVICORN_WORKERS", "UVICORN_WORKERS", "WEB_CONCURRENCY"),
    )
    WORKER_REPLICAS: int | None = Field(
        default=None,
        validation_alias=AliasChoices("WORKER_REPLICAS", "REPLICAS"),
    )
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

    @field_validator("MAX_QUEUE_WAIT", mode="before")
    @classmethod
    def parse_max_queue_wait(cls, v):
        if v is None or isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            try:
                return int(float(v))
            except ValueError:
                return None
        return None

    def admission_settings_snapshot(self) -> dict:
        return AdmissionSettingsSnapshot(
            inflight_limit=self.INFLIGHT_LIMIT,
            max_queue_size=self.MAX_QUEUE_SIZE,
            backpressure_max_queue_delay_ms=int(self.BACKPRESSURE_MAX_QUEUE_DELAY_MS),
            async_throughput_per_sec=self.ASYNC_THROUGHPUT_PER_SEC,
            max_job_age_ms=self.MAX_JOB_AGE_MS,
            max_queue_wait=self.MAX_QUEUE_WAIT,
            worker_count=self.WORKER_COUNT,
            worker_semaphore=self.WORKER_SEMAPHORE,
            embed_batch_size=self.EMBED_BATCH_SIZE,
            embed_batch_timeout_ms=int(self.EMBED_BATCH_TIMEOUT_MS),
            api_uvicorn_workers=self.API_UVICORN_WORKERS,
        ).model_dump()

    def service_runtime_snapshot(
        self,
        service: str,
        *,
        worker_batch_size: int | None = None,
        worker_batch_collect_timeout_ms: int | None = None,
    ) -> dict:
        build_version = self.APP_VERSION or self.IMAGE_TAG or self.GIT_SHA
        return ServiceRuntimeSnapshot(
            service=service,
            build_version=build_version,
            git_sha=self.GIT_SHA,
            image_tag=self.IMAGE_TAG,
            admission_settings=AdmissionSettingsSnapshot(
                inflight_limit=self.INFLIGHT_LIMIT,
                max_queue_size=self.MAX_QUEUE_SIZE,
                backpressure_max_queue_delay_ms=int(self.BACKPRESSURE_MAX_QUEUE_DELAY_MS),
                async_throughput_per_sec=self.ASYNC_THROUGHPUT_PER_SEC,
                max_job_age_ms=self.MAX_JOB_AGE_MS,
                max_queue_wait=self.MAX_QUEUE_WAIT,
                worker_count=self.WORKER_COUNT,
                worker_semaphore=self.WORKER_SEMAPHORE,
                embed_batch_size=self.EMBED_BATCH_SIZE,
                embed_batch_timeout_ms=int(self.EMBED_BATCH_TIMEOUT_MS),
                api_uvicorn_workers=self.API_UVICORN_WORKERS,
            ),
            uvicorn_workers=self.API_UVICORN_WORKERS,
            worker_replicas=self.WORKER_REPLICAS,
            worker_count=self.WORKER_COUNT,
            worker_semaphore=self.WORKER_SEMAPHORE,
            worker_batch_size=worker_batch_size,
            worker_batch_collect_timeout_ms=worker_batch_collect_timeout_ms,
            embed_batch_size=self.EMBED_BATCH_SIZE,
            embed_batch_timeout_ms=int(self.EMBED_BATCH_TIMEOUT_MS),
        ).model_dump()

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

    @property
    def jwt_secret(self) -> str:
        # JWT-секрет: явный JWT_SECRET, иначе fallback на SECRET_KEY.
        # В production предупреждаем, если используется дефолтный SECRET_KEY.
        secret = self.JWT_SECRET or self.SECRET_KEY
        if self.is_production and not self.JWT_SECRET:
            logger.warning(
                "JWT_SECRET не задан — используется дефолтный SECRET_KEY. "
                "Задайте JWT_SECRET в production."
            )
        return secret

    @property
    def api_keys_set(self) -> set[str]:
        return {k.strip() for k in (self.API_KEYS or "").split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
