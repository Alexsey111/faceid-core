import numpy as np

from app.core.celery_app import celery_app
from app.services.search_service import SearchService


@celery_app.task(name="app.tasks.add_embedding")
def add_embedding_task(vector: list, user_id: int):
    try:
        vector_np = np.array(vector, dtype=np.float32)

        if SearchService._faiss_index:
            SearchService._faiss_index.add_one(vector_np, user_id)

    except Exception:
        # не падаем
        pass