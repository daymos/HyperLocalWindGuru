"""
Passphrase-based encryption for archived datasets.

A passphrase (e.g. supplied via the ARCHIVE_KEY env var) is stretched into a
32-byte key with PBKDF2-HMAC-SHA256, then used with Fernet (AES-128-CBC + HMAC,
authenticated). A fresh random salt is generated per file and prepended to the
ciphertext, so the same passphrase yields different files each time and no salt
needs to be stored separately.

Usage:
    from archive_crypto import write_encrypted_parquet, read_encrypted_parquet
    write_encrypted_parquet(df, "data/foo.parquet.enc", os.environ["ARCHIVE_KEY"])
    df = read_encrypted_parquet("data/foo.parquet.enc", os.environ["ARCHIVE_KEY"])
"""

import base64
import io
import os

import pandas as pd
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

_SALT_LEN = 16
_ITERATIONS = 200_000


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=_ITERATIONS)
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))


def encrypt_bytes(data: bytes, passphrase: str) -> bytes:
    salt = os.urandom(_SALT_LEN)
    token = Fernet(_derive_key(passphrase, salt)).encrypt(data)
    return salt + token


def decrypt_bytes(blob: bytes, passphrase: str) -> bytes:
    salt, token = blob[:_SALT_LEN], blob[_SALT_LEN:]
    return Fernet(_derive_key(passphrase, salt)).decrypt(token)


def write_encrypted_parquet(df: pd.DataFrame, path, passphrase: str) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    with open(path, "wb") as f:
        f.write(encrypt_bytes(buf.getvalue(), passphrase))


def read_encrypted_parquet(path, passphrase: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        blob = f.read()
    return pd.read_parquet(io.BytesIO(decrypt_bytes(blob, passphrase)))
