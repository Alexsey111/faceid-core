# app\infrastructure\minio_client.py

import logging

from minio import Minio
from minio.error import S3Error

from app.core.config import settings

logger = logging.getLogger("minio")

# Префикс, под которым живут исходные фото верификаций (удаляются после
# обработки; lifecycle-правило ниже — страховка от «забытых» объектов).
_VERIFY_PREFIX = "verify/"
# S3 Lifecycle поддерживает только целые дни (нет часовой гранулярности),
# поэтому backstop = 1 день. Явный delete_image (в verify_worker,
# _cleanup_minio_image) — основной путь удаления; lifecycle покрывает только
# случай его провала.
_VERIFY_LIFECYCLE_DAYS = 1


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
        found = self.client.bucket_exists(self.bucket)

        if not found:
            self.client.make_bucket(self.bucket)

        # Страховка (ТЗ 5): auto-expire исходных фото верификаций, если явный
        # delete_image не отработал (воркер упал до finalize и т.п.).
        # Best-effort: неудача установки правила не должна ронять приложение.
        self._apply_verify_lifecycle()

    def _apply_verify_lifecycle(self) -> None:
        try:
            from minio.lifecycleconfig import (
                Expiration,
                Filter,
                LifecycleConfig,
                Rule,
            )

            config = LifecycleConfig(
                [
                    Rule(
                        status="Enabled",
                        expiration=Expiration(days=_VERIFY_LIFECYCLE_DAYS),
                        rule_filter=Filter(prefix=_VERIFY_PREFIX),
                        rule_id="expire-verify-images",
                    ),
                ]
            )
            self.client.set_bucket_lifecycle(self.bucket, config)
        except Exception:
            logger.warning(
                "minio_lifecycle_apply_failed bucket=%s prefix=%s",
                self.bucket,
                _VERIFY_PREFIX,
                exc_info=True,
            )

    def upload_image(self, object_name: str, data: bytes, content_type: str = "image/jpeg"):
        from io import BytesIO

        try:
            self.client.put_object(
                bucket_name=self.bucket,
                object_name=object_name,
                data=BytesIO(data),
                length=len(data),
                content_type=content_type,
            )
        except S3Error as e:
            raise RuntimeError(f"MinIO upload error: {e}")

    def get_image(self, object_name: str) -> bytes:
        response = None
        try:
            response = self.client.get_object(self.bucket, object_name)
            return response.read()
        except S3Error as e:
            raise RuntimeError(f"MinIO read error: {e}")
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def delete_image(self, object_name: str):
        try:
            self.client.remove_object(self.bucket, object_name)
        except S3Error as e:
            raise RuntimeError(f"MinIO delete error: {e}")
