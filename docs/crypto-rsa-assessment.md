# Оценка необходимости RSA-2048 — пункт 3 аудита ТЗ

**Контекст.** ТЗ, раздел «Безопасность»: шифрование биометрии —
«AES-256 / RSA-2048». В проекте реализован **AES-256-GCM** (`app/core/crypto.py`).
Документ оценивает: требуется ли дополнительно RSA-2048, или AES-256-GCM
достаточно для соответствия ТЗ и 152-ФЗ.

## Реализация

`app/core/crypto.py` — AES-256-GCM (authenticated encryption):

| Параметр | Значение |
|---|---|
| Алгоритм | AES-256-GCM (симметричный, authenticated) |
| Ключ | 32 байта (`BIOMETRY_AES_KEY_B64`, env-секрет) |
| Nonce | 12 байт, случайный на каждое шифрование |
| Tag | 16 байт (GCM — целостность + аутентификация) |
| Формат | `nonce ‖ tag ‖ ciphertext` |

Применение: эмбеддинги (512D float32) шифруются перед записью в БД
(`embeddings.encrypted_embedding`), расшифровываются при verify.
AES-256-GCM даёт **конфиденциальность + целостность** одновременно —
tamper шифр-текта detect'ится на `decrypt_and_verify` (исключение).

## `encrypted_hash` — content-hash для idempotency/lookup (ТЗ-схема)

ТЗ-схема (CLAUDE.md) требует в `face_embeddings` поле `encrypted_hash TEXT NOT NULL`.
Реализовано (2026-07-07, пункт 8 аудита):

- `app/core/crypto.py:hash_vector` — `sha256(normalized_embedding_bytes).hexdigest()`,
  детерминированный content-hash от plaintext.
- `app/models/embedding.py:encrypted_hash` (Text, NOT NULL, index) — хранится
  отдельно от `encrypted_embedding`.
- `app/db/repositories/embedding_repo.create_embedding` вычисляет hash при enroll.
- Миграция `0003_add_encrypted_hash` (server_default='' для существующих строк).

**Зачем отдельно от GCM-tag:** AES-GCM nonce случаен → `encrypted_embedding`
уникален при каждом шифровании даже для того же вектора → по шифртексту
idempotency не определить. `encrypted_hash` (sha256 от plaintext) даёт
content-based idempotency/lookup (`WHERE encrypted_hash = :h`) без decrypt-all.
GCM-tag внутри `encrypted_embedding` остаётся для integrity при расшифровке.

## ТЗ «AES-256 / RSA-2048» — это опции (OR), не оба

Слэш в формулировке ТЗ — перечень **допустимых** алгоритмов шифрования
биометрии, не требование применить оба. Оба — признанные стандарты для
защиты биометрических данных (152-ФЗ требует «криптографическую защиту»,
не конкретный алгоритм; ФСБ-рекомендованные — и AES-256, и RSA-2048).
**AES-256-GCM выбран — соответствует ТЗ.**

## Почему RSA-2048 не требуется для текущей threat-model

RSA-2048 — **асимметричное** шифрование. Его применение в биометрии:
1. Шифрование публичным ключом получателя (внешний отправитель → сервис).
2. Цифровая подпись (non-repudiation, аутентификация источника).
3. Key wrapping / KMS (гибридная схема: RSA оборачивает AES-ключ).

**Threat-model FaceID Core** (из ТЗ + архитектуры):
- Биометрия поступает по **HTTPS** (транспорт защищён TLS — там RSA/ECDHE
  уже работает на handshake). Сервис **сам** генерирует эмбеддинг и **сам**
  шифрует перед записью — нет сценария «внешний отправитель шифрует
  публичным ключом сервиса». Случай (1) неприменим.
- ТЗ требует **irreversibility + шифрование при хранении**, не цифровую
  подпись / non-repudiation. GCM-tag уже даёт целостность. Случай (2) не
  требуется ТЗ.
- Deploy — Docker Compose on-prem (constraint ТЗ: «Kubernetes опционально»,
  «1-2 разработчика → no overengineering»). KMS key-wrapping — enterprise-
  паттерн, избыточен для on-prem. Случай (3) — overengineering.

**Вывод:** RSA-2048 даёт **нулевой security-выигрыш** для текущей threat-model
и нарушает constraint «no overengineering». AES-256-GCM — достаточен и
соответствует ТЗ.

## Честная фиксация: что RSA-2048 дал бы (если threat-model расширится)

Не реализуется сейчас, но фиксируется как **optional future-hardening**
(отдельные задачи, если появятся соответствующие требования):

| Сценарий | Что даёт RSA-2048 | Когда нужно | Статус |
|---|---|---|---|
| Цифровая подпись эмбеддингов | Non-repudiation; защита от подделки валидного шифр-текста админом БД с доступом к AES-ключу | Требование юридической неотказуемости / multi-party trust | Не требуется ТЗ |
| KMS key-wrapping | AES-ключ обёрнут RSA public-ключом KMS; ротация/аудит ключей без пересшифрования | Cloud/MANAGED KMS, compliance SOC2/ФСТЭК | Overkill для on-prem Compose |
| mTLS client-cert | Взаимная аутентификация сервисов RSA-сертификатами | Zero-trust network, multi-tenant | Транспортная задача (roadmap п.20 HTTPS), отдельная |

## Verdict по ТЗ

- **Соответствие «AES-256 / RSA-2048»: достигнуто.** AES-256-GCM реализован,
  применяется к эмбеддингам, документирован (README, deploy-runbook).
- **RSA-2048: не требуется** (OR-опция, threat-model не требует асимметрии,
  внедрение = overengineering без security-выигрыша).
- **152-ФЗ:** криптографическая защита биометрии обеспечена AES-256-GCM
  (конфиденциальность + целостность at-rest); исходные фото не хранятся
  (удаляются после извлечения эмбеддинга); логи без биометрии.

**Действий по коду не требуется.** Документ — обоснование выбора для
аудита ТЗ.