# anti_replay_service.py - Защита от replay-атак

import hashlib

from app.infrastructure.redis_client import redis_client


class AntiReplayService:
    """
    Защита от повторных атак (replay attacks).
    Использует Redis для хранения хешей изображений.
    """

    @staticmethod
    def check(image_bytes: bytes, ttl: int = 10) -> bool:
        """
        Проверяет, было ли это изображение недавно обработано.

        Args:
            image_bytes: Байты изображения
            ttl: Время жизни записи в секундах (default: 10 сек)

        Returns:
            bool: True = OK (уникальное), False = replay detected
        """
        # Вычисляем хеш изображения
        h = hashlib.sha256(image_bytes).hexdigest()

        # Проверяем наличие в Redis
        exists = redis_client.get(h)
        if exists:
            return False

        # Сохраняем хеш с TTL
        redis_client.set(h, "1", ttl=ttl)
        return True

    @staticmethod
    def check_with_hash(image_hash: str, ttl: int = 10) -> bool:
        """
        Проверяет по готовому хешу.

        Args:
            image_hash: SHA256 хеш изображения
            ttl: Время жизни записи в секундах

        Returns:
            bool: True = OK, False = replay detected
        """
        exists = redis_client.get(image_hash)
        if exists:
            return False

        redis_client.set(image_hash, "1", ttl=ttl)
        return True

    @staticmethod
    def compute_hash(image_bytes: bytes) -> str:
        """
        Вычисляет SHA256 хеш изображения.

        Args:
            image_bytes: Байты изображения

        Returns:
            str: Hex-строка хеша
        """
        return hashlib.sha256(image_bytes).hexdigest()
