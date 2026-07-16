#!/usr/bin/env python3
# scripts/gen_aes_key.py — генерация 256-битного AES-ключа для шифрования эмбеддингов.
#
# 152-ФЗ: эмбеддинги лиц шифруются AES-256-GCM (app/core/crypto.py).
# Мастер-ключ хранится в BIOMETRY_AES_KEY_B64 (.env.prod, вне git). В production
# без ключа сервис падает при старте (config.py: aes_key → RuntimeError, fail-closed).
#
# Запуск:
#   python scripts/gen_aes_key.py
#   → выводит base64-строку 32 байт (256 бит), вставить в .env.prod:
#     BIOMETRY_AES_KEY_B64=<вывод>
#
# Ротация: перевыпуск ключа требует перешифрования всех эмбеддингов
# (decrypt старым → encrypt новым) — см. docs/deploy-runbook.md (раздел key rotation).

import base64
import os
import sys


def main() -> int:
    key = os.urandom(32)  # 256 бит — AES-256
    encoded = base64.b64encode(key).decode("ascii")
    # Длина base64(32 байта) = 44 символа, последний '=' — padding, обязателен.
    print(encoded)
    print(
        "\nВставьте это значение в .env.prod как BIOMETRY_AES_KEY_B64.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())