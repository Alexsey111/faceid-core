import hashlib

import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

from app.core.config import settings


def _get_key() -> bytes:
    key = settings.aes_key
    if len(key) != 32:
        raise ValueError("BIOMETRY_AES_KEY_B64 must decode to 32 bytes")
    return key


def _encrypt(data: bytes) -> bytes:
    key = _get_key()
    nonce = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return nonce + tag + ciphertext


def _decrypt(payload: bytes) -> bytes:
    key = _get_key()
    nonce = payload[:12]
    tag = payload[12:28]
    ciphertext = payload[28:]
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)


def encrypt(data: bytes) -> bytes:
    return _encrypt(data)


def decrypt(data: bytes) -> bytes:
    return _decrypt(data)


def encrypt_embedding_bytes(data: bytes) -> bytes:
    return encrypt(data)


def decrypt_embedding_bytes(token: bytes) -> bytes:
    return decrypt(token)


def encrypt_vector(vector) -> bytes:
    if isinstance(vector, (bytes, bytearray, memoryview)):
        return encrypt(bytes(vector))

    normalized = vector.astype(np.float32, copy=False)
    assert normalized.dtype == np.float32
    return encrypt(normalized.tobytes())


def decrypt_vector(data: bytes) -> np.ndarray:
    decrypted = decrypt(data)
    return np.frombuffer(decrypted, dtype=np.float32)


def hash_vector(vector) -> str:
    """Content-hash нормированного эмбеддинга (sha256 от plaintext-байтов).

    Хранится отдельно от encrypted_embedding (в колонке encrypted_hash) — даёт
    content-based idempotency/lookup (`WHERE encrypted_hash = :h`) без decrypt-all.
    ТЗ-схема (CLAUDE.md) требует encrypted_hash TEXT NOT NULL в face_embeddings.
    Hash от plaintext: одинаковый вектор → одинаковый hash (детерминированно);
    AES-GCM nonce случайный → encrypted_embedding уникален даже для того же вектора,
    поэтому по шифртексту idempotency не определить — отдельный hash необходим.
    """
    if isinstance(vector, (bytes, bytearray, memoryview)):
        data = bytes(vector)
    else:
        data = np.asarray(vector, dtype=np.float32).tobytes()
    return hashlib.sha256(data).hexdigest()
