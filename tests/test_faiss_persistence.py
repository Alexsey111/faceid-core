import os
import numpy as np
import pytest

from app.services.faiss_index import FaissIndex
from app.core.config import settings


@pytest.fixture
def clean_files():
    """Удаляем файлы перед тестом"""
    for f in ["faiss.index", "faiss_meta.pkl"]:
        if os.path.exists(f):
            os.remove(f)
    yield
    for f in ["faiss.index", "faiss_meta.pkl"]:
        if os.path.exists(f):
            os.remove(f)


def random_vec():
    v = np.random.rand(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_faiss_save_and_load(clean_files):
    settings.FAISS_PERSIST_ENABLED = True

    index = FaissIndex()

    v1 = random_vec()
    v2 = random_vec()

    index.add_one(v1, user_id=1)
    index.add_one(v2, user_id=2)

    # новый объект → должен загрузить с диска
    index2 = FaissIndex()

    assert index2.index.ntotal == 2
    assert len(index2.user_ids) == 2


def test_faiss_search_after_reload(clean_files):
    settings.FAISS_PERSIST_ENABLED = True

    index = FaissIndex()

    v1 = random_vec()
    v2 = random_vec()

    index.add_one(v1, user_id=1)
    index.add_one(v2, user_id=2)

    # reload
    index2 = FaissIndex()

    result = index2.search(v1, k=1)

    assert result
    assert result[0]["user_id"] == 1
    assert result[0]["similarity"] > 0.9


def test_faiss_rebuild_consistency(clean_files):
    settings.FAISS_PERSIST_ENABLED = True

    index = FaissIndex()

    items = [
        {"user_id": 1, "embedding": random_vec()},
        {"user_id": 2, "embedding": random_vec()},
    ]

    index.rebuild(items)

    result = index.search(items[0]["embedding"], k=1)

    assert result[0]["user_id"] == 1
    assert result[0]["similarity"] > 0.9


def test_faiss_rebuild_empty(clean_files):
    index = FaissIndex()

    index.add_one(random_vec(), user_id=1)
    assert index.index.ntotal == 1

    index.rebuild([])

    assert index.index.ntotal == 0
    assert index.user_ids == []