"""add request_ip column to verification_logs

Revision ID: 0004_add_request_ip
Revises: 0003_add_encrypted_hash
Create Date: 2026-07-16

ТЗ-схема (CLAUDE.md) требует в verification_logs поле `request_ip INET` —
audit-лог клиента для расследования инцидентов (кто пытался верифицироваться,
откуда; 152-ФЗ ПДн-аудит). Audit E2 (tz-code-auditor 2026-07-16): nginx ставит
X-Real-IP/X-Forwarded-For (infrastructure/nginx/api_lb.conf), но они доходили
только до rate_limiter, в VerificationLog не записывались.

Хранение: String(45) вместо postgres-only INET — переносимость + IP как строка
удобна для логов/отладки; валидация формата при извлечении (extract_client_ip),
не на уровне БД. IPv4 (≤15) / IPv6 (≤39) вмещается в 45.

Проброс: через contextvar (app/core/request_context.py) — middleware ставит IP
на каждый HTTP-запрос, create_log читает. Background-логирование через
asyncio.create_task копирует context → IP доезжает. Worker-путь: IP в
job-payload + restore в contextvar (см. verify_worker).

Для существующих строк: nullable=True (старые записи без IP, новые пишут).
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0004_add_request_ip"
down_revision = "0003_add_encrypted_hash"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "verification_logs",
        sa.Column("request_ip", sa.String(length=45), nullable=True),
    )
    # Индекс для audit-запросов «все верификации с этого IP» (расследование
    # инцидентов / выявление brute-force по одному источнику).
    op.create_index(
        op.f("ix_verification_logs_request_ip"),
        "verification_logs",
        ["request_ip"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_verification_logs_request_ip"), table_name="verification_logs")
    op.drop_column("verification_logs", "request_ip")