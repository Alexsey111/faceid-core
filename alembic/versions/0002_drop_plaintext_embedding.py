"""drop plaintext embedding column

Revision ID: 0002_drop_plaintext_embedding
Revises: 0001_initial_schema
Create Date: 2026-06-30

Убираем plaintext-копию эмбеддинга (pgvector-колонка `embedding`).
Биометрия теперь хранится ТОЛЬКО зашифрованной в `encrypted_embedding` (AES-256-GCM),
что соответствует ТЗ (раздел 5: «биометрические данные — обязательное шифрование»).
Поиск идёт через FAISS (строится из encrypted при старте); SQL-fallback переписан
на decrypt-all + numpy-cosine в EmbeddingRepository.

Безопасно: `encrypted_embedding` — NOT NULL с миграции 0001, create_embedding
всегда писал обе колонки → данных достаточно, backfill не требуется.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision = "0002_drop_plaintext_embedding"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("embeddings", "embedding")


def downgrade() -> None:
    op.add_column(
        "embeddings",
        sa.Column("embedding", Vector(512), nullable=True),
    )