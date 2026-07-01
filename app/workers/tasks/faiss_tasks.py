# app\workers\tasks\faiss_tasks.py

import logging

import numpy as np

from app.workers.celery_app import celery_app
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)


@celery_app.task(
    name="app.workers.tasks.add_embedding",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
    max_retries=5,
)
def add_embedding_task(self, vector: list, user_id: int):
    import numpy as np

    try:
        vector_np = np.array(vector, dtype=np.float32)

        if SearchService._faiss_index:
            SearchService._faiss_index.add_one(vector_np, user_id)

    except Exception as e:
        logger.exception("FAISS task failed", extra={"user_id": user_id})
        raise self.retry(exc=e)


@celery_app.task(
    name="app.workers.tasks.add_embeddings_batch",
    bind=False,
)
def add_embeddings_batch(vectors: list, user_ids: list):
    import numpy as np

    try:
        vectors_np = np.array(vectors, dtype=np.float32)

        if SearchService._faiss_index:
            SearchService._faiss_index.add(vectors_np, user_ids)

    except Exception:
        pass
