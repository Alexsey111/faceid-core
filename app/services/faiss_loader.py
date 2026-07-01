# app/services/faiss_loader.py

import numpy as np
from app.services.faiss_index import FaissIndex
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.core.crypto import decrypt_vector


async def build_faiss_index(repo: EmbeddingRepository) -> FaissIndex:
    users = await repo.get_all_users_with_embeddings()

    if not users:
        return FaissIndex()

    vectors = []
    user_ids = []

    for user in users:
        for emb in user.embeddings:
            if not emb.encrypted_embedding:
                continue
            vector = decrypt_vector(emb.encrypted_embedding)
            vectors.append(vector)
            user_ids.append(user.id)

    if not vectors:
        return FaissIndex()

    vectors = np.stack(vectors)

    index = FaissIndex()
    index.add(vectors, user_ids)

    return index
