#!/usr/bin/env bash
# Генерация self-signed TLS-сертификата для dev-окружения FaceID Core.
# prod: замените содержимое ./certs/ на реальные сертификаты (LE/внутренний CA).
#
# Запуск из корня репозитория (там, где docker-compose.yml):
#   bash infrastructure/nginx/generate_self_signed.sh
set -euo pipefail

CERTS_DIR="./certs"
CERT="${CERTS_DIR}/cert.pem"
KEY="${CERTS_DIR}/key.pem"

mkdir -p "${CERTS_DIR}"

if [[ -f "${CERT}" && -f "${KEY}" && "${FORCE:-0}" != "1" ]]; then
  echo "[generate_self_signed] Сертификаты уже существуют в ${CERTS_DIR}/ (FORCE=1 для перегенерации)."
  exit 0
fi

echo "[generate_self_signed] Генерирую RSA 2048 self-signed (CN=localhost, 365 дней)..."
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout "${KEY}" \
  -out "${CERT}" \
  -days 365 \
  -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

chmod 600 "${KEY}"
echo "[generate_self_signed] Готово: ${CERT}, ${KEY}"
echo "[generate_self_signed] Подключается в docker-compose (api_lb.volumes ./certs:/etc/nginx/ssl:ro)."