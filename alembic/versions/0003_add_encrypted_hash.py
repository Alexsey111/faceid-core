"""add encrypted_hash column

Revision ID: 0003_add_encrypted_hash
Revises: 0002_drop_plaintext_embedding
Create Date: 2026-07-07

ТЗ-схема (CLAUDE.md) требует в face_embeddings поле `encrypted_hash TEXT NOT NULL`.
Раньше колонки не было (lookup по user_id, integrity через AES-GCM tag внутри
encrypted_embedding). Колонка добавляет content-based idempotency/lookup
(`WHERE encrypted_hash = :h`) без decrypt-all: AES-GCM nonce случаен →
encrypted_embedding уникален даже для того же вектора, поэтому по шифртексту
idempotency не определить — отдельный sha256-hash от plaintext необходим.

Hash = sha256(normalized_embedding_bytes).hex() (app.core.crypto.hash_vector),
вычисляется при enroll (embedding_repo.create_embedding).

Для существующих строк: backfill невозможен в миграции (нужен decrypt + key),
поэтому server_default='' — старые записи получают пустой hash, новые enroll
заполняют реальным. NOT NULL (пустая строка != NULL). Для полной консистентности
существующих данных — re-enroll через /update-reference.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0003_add_encrypted_hash"
down_revision = "0002_drop_plaintext_embedding"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # server_default='' чтобы миграция прошла на существующих данных (hash
    # неизвестен без decrypt+key); новые enroll заполнят реальным hash.
    op.add_column(
        "embeddings",
        sa.Column("encrypted_hash", sa.Text(), nullable=False, server_default=""),
    )
    op.create_index(
        op.f("ix_embeddings_encrypted_hash"), "embeddings", ["encrypted_hash"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_embeddings_encrypted_hash"), table_name="embeddings")
    op.drop_column("embeddings", "encrypted_hash")