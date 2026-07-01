# anti_replay_service.py - Защита от replay-атак

import hashlib
from time import perf_counter

from app.infrastructure.redis_client import redis_client
from app.monitoring.metrics import REDIS_COMMAND_LATENCY_MS


class AntiReplayService:
    """
    Защита от повторных атак (replay attacks).
    Использует Redis для хранения хешей изображений.
    """

    @staticmethod
    def check(image_bytes: bytes, ttl: int = 10) -> bool:
        image_hash = AntiReplayService.compute_hash(image_bytes)
        return AntiReplayService.check_with_hash(image_hash, ttl=ttl)

    @staticmethod
    def check_with_hash(image_hash: str, ttl: int = 10) -> bool:
        start = perf_counter()
        added = redis_client.set_if_absent(image_hash, "1", ttl=ttl)
        REDIS_COMMAND_LATENCY_MS.labels(command="anti_replay_set_nx_ex").observe(
            (perf_counter() - start) * 1000.0
        )
        return added

    @staticmethod
    def compute_hash(image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()