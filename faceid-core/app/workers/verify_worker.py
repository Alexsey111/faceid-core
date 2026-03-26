# app/workers/verify_worker.py

import json
import base64
import asyncio
import redis
from typing import Tuple, cast

from app.db.session import AsyncSessionLocal
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.verification_repo import VerificationRepository
from app.services.search_service import SearchService
from app.services.verify_result_store import VerifyResultStore
from app.services.verification_service import VerificationService

redis_client = redis.Redis(host="redis", port=6379, db=0)
QUEUE_NAME = "face_verify_queue"
semaphore = asyncio.Semaphore(2)


async def process_job(job_data: dict):
    async with semaphore:
        job_id = job_data["job_id"]
        payload = job_data["payload"]

        try:
            image_bytes = base64.b64decode(payload["image_b64"])

            async with AsyncSessionLocal() as db:
                embedding_repo = EmbeddingRepository(db)
                verification_repo = VerificationRepository(db)
                search_service = SearchService(embedding_repo)

                service = VerificationService(
                    embedding_repo=embedding_repo,
                    verification_repo=verification_repo,
                    search_service=search_service,
                    load_pipeline=True,
                )

                result = await service.verify_face(
                    image_bytes=image_bytes,
                    user_id=payload.get("user_id"),
                    require_liveness=payload.get("require_liveness", False),
                    job_id=job_id,
                )

            VerifyResultStore.set_done(job_id, result)

        except Exception as e:
            VerifyResultStore.set_error(job_id, str(e))


async def run_worker():
    while True:
        job = cast(
            Tuple[bytes, bytes] | None,
            redis_client.blpop([QUEUE_NAME])
        )

        if not job:
            continue

        _, data = job
        job_data = json.loads(data.decode("utf-8"))

        await process_job(job_data)


if __name__ == "__main__":
    asyncio.run(run_worker())
