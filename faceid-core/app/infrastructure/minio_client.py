# faceid-core\app\infrastructure\minio_client.py

from minio import Minio
from minio.error import S3Error
from app.core.config import settings


class MinioClient:

    def __init__(self):
        self.client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE
        )

        self.bucket = settings.MINIO_BUCKET

        self._ensure_bucket()

    def _ensure_bucket(self):
        """Создает bucket если его нет"""
        found = self.client.bucket_exists(self.bucket)

        if not found:
            self.client.make_bucket(self.bucket)

    def upload_image(self, object_name: str, data: bytes):
        """Загрузка изображения"""

        from io import BytesIO

        try:
            self.client.put_object(
                bucket_name=self.bucket,
                object_name=object_name,
                data=BytesIO(data),
                length=len(data),
                content_type="image/jpeg"
            )

        except S3Error as e:
            raise RuntimeError(f"MinIO upload error: {e}")

    def get_image(self, object_name: str) -> bytes:
        """Получить изображение"""

        try:
            response = self.client.get_object(self.bucket, object_name)
            return response.read()

        except S3Error as e:
            raise RuntimeError(f"MinIO read error: {e}")

    def delete_image(self, object_name: str):
        """Удалить изображение"""

        try:
            self.client.remove_object(self.bucket, object_name)

        except S3Error as e:
            raise RuntimeError(f"MinIO delete error: {e}")