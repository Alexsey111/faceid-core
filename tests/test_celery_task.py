import numpy as np

from app.services.search_service import SearchService
from app.services.faiss_index import FaissIndex
from app.tasks.faiss_tasks import add_embedding_task


def test_add_embedding_task():
    SearchService._faiss_index = FaissIndex()

    vector = np.random.rand(512).astype("float32")

    add_embedding_task(vector.tolist(), user_id=123)

    assert SearchService._faiss_index.index.ntotal == 1