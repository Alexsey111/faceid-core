"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-03-25
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import UUID


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("external_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index(op.f("ix_users_external_id"), "users", ["external_id"], unique=True)

    op.create_table(
        "verification_jobs",
        sa.Column("id", UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column(
            "status",
            sa.Enum("pending", "processing", "done", "failed", name="job_status", native_enum=False),
            nullable=False,
        ),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index(op.f("ix_verification_jobs_status"), "verification_jobs", ["status"], unique=False)

    op.create_table(
        "verification_logs",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("similarity", sa.Float(), nullable=True),
        sa.Column("margin", sa.Float(), nullable=True),
        sa.Column("liveness_score", sa.Float(), nullable=True),
        sa.Column("is_genuine", sa.Boolean(), nullable=True),
        sa.Column("result", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index(op.f("ix_verification_logs_user_id"), "verification_logs", ["user_id"], unique=False)

    op.create_table(
        "embeddings",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("embedding", Vector(512), nullable=True),
        sa.Column("encrypted_embedding", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index(op.f("ix_embeddings_user_id"), "embeddings", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_embeddings_user_id"), table_name="embeddings")
    op.drop_table("embeddings")

    op.drop_index(op.f("ix_verification_logs_user_id"), table_name="verification_logs")
    op.drop_table("verification_logs")

    op.drop_index(op.f("ix_verification_jobs_status"), table_name="verification_jobs")
    op.drop_table("verification_jobs")

    op.drop_index(op.f("ix_users_external_id"), table_name="users")
    op.drop_table("users")

    op.execute("DROP EXTENSION IF EXISTS vector")
