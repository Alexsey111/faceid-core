#!/usr/bin/env bash
# scripts/backup_db.sh — резервное копирование БД FaceID Core (152-ФЗ).
#
# Дамп содержит PII (users.external_id) и зашифрованные эмбеддинги (AES-256-GCM
# в БД — ciphertext, не plaintext). Дамп — КОНФИДЕНЦИАЛЬНЫЙ: хранить в защищённом
# месте, не рядом с открытым кодом, ротировать. Не загружать в git.
#
# Запуск (с уже поднятым стеком prod):
#   bash scripts/backup_db.sh
#   bash scripts/backup_db.sh /var/backups/faceid 14   # каталог, хранить N дней
#
# Cron (хост, ежедневно 03:17):
#   17 3 * * * cd /opt/faceid-core && bash scripts/backup_db.sh /var/backups/faceid 14 >> /var/log/faceid-backup.log 2>&1
#
# Восстановление:
#   gunzip -c /var/backups/faceid/faceid_YYYYMMDD_HHMMSS.sql.gz | \
#     docker compose --env-file .env.prod -f docker-compose.yml -f docker-compose.prod.yml \
#       exec -T postgres psql -U postgres -d faceid
# (alembic-схема восстанавливается из дампа; при пустой БД — сначала alembic upgrade head)

set -euo pipefail

BACKUP_DIR="${1:-./backups}"
KEEP_DAYS="${2:-14}"

# Имя стека/каталога compose — для exec в правильный postgres-контейнер.
COMPOSE_FILES="-f docker-compose.yml -f docker-compose.prod.yml"
ENV_FILE="--env-file .env.prod"

mkdir -p "$BACKUP_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
FILE="$BACKUP_DIR/faceid_${STAMP}.sql.gz"

# pg_dump внутри postgres-контейнера (не зависит от pg_dump на хосте).
# -T = отключить TTY (pipe), обязательно для non-interactive exec в pipeline.
docker compose $ENV_FILE $COMPOSE_FILES exec -T postgres \
  pg_dump -U postgres -d faceid --no-owner --no-privileges \
  | gzip -9 > "$FILE"

# Ротация: удалять старше KEEP_DAYS дней.
find "$BACKUP_DIR" -name 'faceid_*.sql.gz' -type f -mtime "+${KEEP_DAYS}" -delete

SIZE="$(du -h "$FILE" | cut -f1)"
echo "[$(date -Iseconds)] backup OK: $FILE ($SIZE), retention=${KEEP_DAYS}d"