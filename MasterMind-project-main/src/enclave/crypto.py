"""crypto.py — AES-256-GCM primitives for the Credential Enclave.

This module is deliberately *pure*: no I/O, no environment access, no
logging of any payload. Every function operates on bytes in, bytes out.
Higher layers (storage.py, enclave.py) compose these primitives.

Security invariants enforced here
---------------------------------
1. AES-256-GCM only. 32-byte keys, 12-byte nonces (NIST SP 800-38D).
2. A fresh random nonce per encryption — never reused, never caller-supplied.
3. AAD binding: every ciphertext is bound to its logical credential name,
   so a blob for ``OPENAI_API_KEY`` cannot be replayed as ``TWITTER_API_SECRET``
   even by an attacker with full write access to the database.
4. Constant-time tag verification (delegated to the `cryptography` library's
   OpenSSL backend).
5. Explicit memory zeroing helpers (`zero_buffer`, `SecretBuffer`) so
   plaintext lives on the heap for the shortest possible window.

What this module can NOT guarantee (documented honestly)
--------------------------------------------------------
CPython strings are immutable and may be interned; once a secret is
converted to `str` for an SDK call, that copy cannot be zeroed. The
mitigation strategy is:
  * keep secrets as `bytearray` for as long as possible,
  * scope the `str` conversion to the narrowest possible frame,
  * zero the backing bytearray in a `finally` block,
  * never store the str on any object attribute.
For HSM-grade guarantees, move decryption into a separate enclave process.
"""

from __future__ import annotations

import base64
import ctypes
import hmac
import os
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_SIZE = 32     # AES-256
NONCE_SIZE = 12   # 96-bit, NIST-recommended for GCM
TAG_SIZE = 16     # 128-bit GCM tag (appended to ciphertext by AESGCM)


class DecryptionError(Exception):
    """GCM authentication failed: wrong key, tampered ciphertext, or wrong AAD.

    Deliberately carries NO detail about which of the three failed —
    distinguishing them would create a padding-oracle-style side channel.
    """


class KeyFormatError(ValueError):
    """The provided master key is not exactly 32 raw bytes."""


# ---------------------------------------------------------------------------
# Memory hygiene
# ---------------------------------------------------------------------------
def zero_buffer(buf: bytearray) -> None:
    """Overwrite *buf* with zeros via ctypes.memset.

    A plain ``buf[:] = b"\\x00" * len(buf)`` allocates a new bytes object
    containing zeros and may be optimised away; memset writes through a raw
    pointer and cannot be elided by the interpreter.
    """
    if not buf:
        return
    addr = ctypes.addressof((ctypes.c_char * len(buf)).from_buffer(buf))
    ctypes.memset(addr, 0, len(buf))


class SecretBuffer:
    """Context manager owning a mutable secret; zeroes itself on exit.

    Usage::

        with SecretBuffer(decrypt(...)) as secret:
            do_api_call(secret.decode())
        # secret bytes are zeroed here, even if do_api_call raised
    """

    __slots__ = ("_buf",)

    def __init__(self, initial: bytes | bytearray) -> None:
        self._buf = bytearray(initial)
        if isinstance(initial, bytearray):
            zero_buffer(initial)  # take ownership; wipe the caller's copy

    def __enter__(self) -> bytearray:
        return self._buf

    def __exit__(self, *exc_info: object) -> None:
        zero_buffer(self._buf)

    def __repr__(self) -> str:  # never leak contents via repr/logging
        return f"<SecretBuffer len={len(self._buf)} REDACTED>"


# ---------------------------------------------------------------------------
# Key handling
# ---------------------------------------------------------------------------
def generate_master_key() -> bytes:
    """Generate a fresh 32-byte AES-256 master key from the OS CSPRNG."""
    return os.urandom(KEY_SIZE)


def load_master_key(b64: str) -> bytes:
    """Decode and validate a base64url-encoded master key."""
    try:
        raw = base64.urlsafe_b64decode(b64)
    except Exception as exc:  # binascii.Error subclasses vary by version
        raise KeyFormatError("Master key is not valid base64url.") from exc
    if len(raw) != KEY_SIZE:
        raise KeyFormatError(f"Master key must be {KEY_SIZE} bytes; got {len(raw)}.")
    return raw


def keys_equal(a: bytes, b: bytes) -> bool:
    """Constant-time comparison for key fingerprint checks."""
    return hmac.compare_digest(a, b)


# ---------------------------------------------------------------------------
# Sealed blob value type
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SealedBlob:
    """An AES-256-GCM sealed credential, safe to persist anywhere.

    Fields map 1:1 to database columns (see storage.py):
      aad        — logical credential name this blob is cryptographically bound to
      nonce      — 12 random bytes, unique per encryption
      ciphertext — ciphertext || 16-byte GCM tag
    """

    aad: str
    nonce: bytes
    ciphertext: bytes

    def __repr__(self) -> str:  # never leak ciphertext into logs
        return f"<SealedBlob aad={self.aad!r} nonce=... ct[{len(self.ciphertext)}B]>"


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------
def seal(master_key: bytes, aad: str, plaintext: bytes) -> SealedBlob:
    """Encrypt *plaintext* under *master_key*, bound to logical name *aad*.

    Raises KeyFormatError if the key is malformed. Never raises on plaintext
    content — empty secrets are legal (some providers use empty tokens in dev).
    """
    if len(master_key) != KEY_SIZE:
        raise KeyFormatError(f"master_key must be {KEY_SIZE} bytes; got {len(master_key)}.")
    nonce = os.urandom(NONCE_SIZE)
    ct = AESGCM(master_key).encrypt(nonce, plaintext, aad.encode())
    return SealedBlob(aad=aad, nonce=nonce, ciphertext=ct)


def unseal(master_key: bytes, blob: SealedBlob) -> bytearray:
    """Decrypt *blob*; returns a mutable bytearray the CALLER MUST zero.

    Prefer wrapping the result in SecretBuffer::

        with SecretBuffer(unseal(key, blob)) as secret:
            ...

    Raises DecryptionError on any authentication failure, with no detail
    about the failure mode (by design).
    """
    if len(master_key) != KEY_SIZE:
        raise KeyFormatError(f"master_key must be {KEY_SIZE} bytes; got {len(master_key)}.")
    try:
        pt = AESGCM(master_key).decrypt(blob.nonce, blob.ciphertext, blob.aad.encode())
    except InvalidTag:
        raise DecryptionError(
            f"Authentication failed for credential {blob.aad!r}."
        ) from None  # `from None` — never chain internals that could leak state
    buf = bytearray(pt)
    # `pt` is an immutable bytes object; we cannot zero it, but it goes out
    # of scope immediately.  The bytearray copy is the caller's to manage.
    del pt
    return buf


def rewrap(old_key: bytes, new_key: bytes, blob: SealedBlob) -> SealedBlob:
    """Key rotation: decrypt under *old_key*, re-encrypt under *new_key*.

    The plaintext exists only inside this frame and is zeroed before return.
    Used by storage.py's `rotate_master_key` bulk operation.
    """
    buf = unseal(old_key, blob)
    try:
        return seal(new_key, blob.aad, bytes(buf))
    finally:
        zero_buffer(buf)
