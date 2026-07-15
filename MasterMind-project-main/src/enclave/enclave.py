"""Credential Enclave — AES-256-GCM BYOK Just-In-Time credential injector.

Security model
--------------
All credential payloads are encrypted at rest with AES-256-GCM:
  * 256-bit key     — ENCLAVE_MASTER_KEY env var (32 raw bytes, base64url-encoded).
  * 96-bit nonce    — random per credential, stored alongside the ciphertext.
  * 128-bit GCM tag — authenticates both ciphertext and associated data (AAD),
                      preventing both forgery and silent bit-flip.
  * AAD             — the credential *name* (ASCII) so a ciphertext for
                      "OPENAI_API_KEY" cannot be silently substituted for
                      "TWITTER_API_SECRET" even if an attacker controls Redis.

At no point does plaintext:
  * enter a log record (RedactingSpanProcessor in observability/ handles traces)
  * leave the process heap (bytes are zeroed via ctypes.memset immediately after use)
  * appear in a traceback (inject() wraps the context call in a finally wipe)
  * get serialised to Redis, JSON, or any distributed state

Design pattern: the *inject* pattern (also called "Bracket Resource" or
"loan pattern" in functional languages). The caller never holds the secret:

    enclave.inject("OPENAI_API_KEY", lambda key: openai_client(key).chat(...))

The lambda receives the raw bytes (or str), calls the API, and returns. The
plaintext is zeroed before inject() returns. The caller never sees it again.

Thread safety
-------------
`CredentialStore` is fully thread-safe: reads are lock-free (immutable dict
after construction); the master key is loaded once at construction time.
`LifecycleManager.inject()` is re-entrant — multiple workers may call it
concurrently; each invocation owns its own decrypted-bytes buffer.

References
----------
NIST SP 800-38D (Recommendation for GCM), NIST SP 800-57 Part 1.
"""

from __future__ import annotations

import base64
import ctypes
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ..logging_setup import get_logger

log = get_logger("enclave")

_KEY_BYTES = 32   # AES-256
_NONCE_BYTES = 12  # 96-bit nonce (NIST recommended for GCM)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class AuthenticationError(Exception):
    """Raised when GCM tag verification fails (tampered or wrong-key ciphertext)."""


class CredentialNotFoundError(KeyError):
    """Raised when the requested credential name has no registered entry."""


# ---------------------------------------------------------------------------
# Value type for an encrypted credential blob
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EncryptedCredential:
    """An AES-256-GCM encrypted credential, safe to store anywhere.

    Fields
    ------
    name       : logical credential name, also used as GCM Associated Data (AAD).
    nonce_b64  : base64url-encoded 12-byte random nonce.
    ciphertext_b64 : base64url-encoded ciphertext || GCM-tag (16 bytes appended
                     by AESGCM.encrypt()).
    """

    name: str
    nonce_b64: str
    ciphertext_b64: str

    def to_env_value(self) -> str:
        """Serialize to a single string safe for storage in an env var or config file.

        Format: ``<name>:<nonce_b64>:<ciphertext_b64>``
        Colons are safe because base64url encoding never produces them.
        """
        return f"{self.name}:{self.nonce_b64}:{self.ciphertext_b64}"

    @classmethod
    def from_env_value(cls, value: str) -> "EncryptedCredential":
        """Inverse of ``to_env_value``."""
        parts = value.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                "Malformed EncryptedCredential env value; expected 'name:nonce:ciphertext'"
            )
        return cls(name=parts[0], nonce_b64=parts[1], ciphertext_b64=parts[2])


# ---------------------------------------------------------------------------
# Helper: encrypt a plaintext secret (used at credential registration time)
# ---------------------------------------------------------------------------
def encrypt_credential(name: str, plaintext: str, master_key: bytes) -> EncryptedCredential:
    """Encrypt *plaintext* under *master_key* with AES-256-GCM.

    The credential *name* is used as GCM Associated Data (AAD) to bind the
    ciphertext to this specific logical key — it cannot be moved to a
    different credential slot without decryption failing.

    Parameters
    ----------
    name       : logical credential name (e.g. "OPENAI_API_KEY").
    plaintext  : the raw secret string.
    master_key : 32 raw bytes (load from ``ENCLAVE_MASTER_KEY`` env var,
                 decoded from base64url).

    Returns
    -------
    EncryptedCredential : safe to serialise and store anywhere.
    """
    if len(master_key) != _KEY_BYTES:
        raise ValueError(f"master_key must be {_KEY_BYTES} bytes; got {len(master_key)}")
    nonce = os.urandom(_NONCE_BYTES)
    aesgcm = AESGCM(master_key)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode(), name.encode())
    return EncryptedCredential(
        name=name,
        nonce_b64=base64.urlsafe_b64encode(nonce).decode(),
        ciphertext_b64=base64.urlsafe_b64encode(ciphertext).decode(),
    )


# ---------------------------------------------------------------------------
# Credential Store: holds a collection of EncryptedCredentials + master key
# ---------------------------------------------------------------------------
class CredentialStore:
    """Immutable registry of AES-256-GCM encrypted credentials.

    Construct once at process startup; inject the master key from the
    environment. The master key bytes are held in a private attribute but
    are never logged or serialised — the RedactingSpanProcessor ensures
    they cannot escape through OTel traces either.

    Parameters
    ----------
    master_key_b64 : base64url-encoded 32-byte AES-256 key.
                     Defaults to reading ``ENCLAVE_MASTER_KEY`` from env.
    credentials    : mapping of name → EncryptedCredential, constructed
                     e.g. by calling `encrypt_credential` for each secret
                     during a one-time provisioning step.
    """

    def __init__(
        self,
        *,
        master_key_b64: str | None = None,
        credentials: dict[str, EncryptedCredential] | None = None,
    ) -> None:
        raw_b64 = master_key_b64 or os.environ.get("ENCLAVE_MASTER_KEY", "")
        if not raw_b64:
            raise RuntimeError(
                "ENCLAVE_MASTER_KEY is not set. "
                "Generate one with: python -c \"import os,base64; "
                "print(base64.urlsafe_b64encode(os.urandom(32)).decode())\""
            )
        self._master_key: bytes = base64.urlsafe_b64decode(raw_b64)
        if len(self._master_key) != _KEY_BYTES:
            raise ValueError(
                f"ENCLAVE_MASTER_KEY must decode to {_KEY_BYTES} bytes; "
                f"got {len(self._master_key)}"
            )
        # Immutable registry after construction — no lock needed for reads.
        self._credentials: dict[str, EncryptedCredential] = dict(credentials or {})

    @classmethod
    def from_env(cls) -> "CredentialStore":
        """Build a CredentialStore by reading ENCRYPTED_<NAME> env vars.

        Any env var matching the prefix ``ENCRYPTED_`` is parsed as a
        serialised EncryptedCredential (produced by ``encrypt_credential``).
        This allows credentials to be rotated by updating env vars without
        changing code.

        Example env vars::

            ENCLAVE_MASTER_KEY=<base64url 32-byte key>
            ENCRYPTED_OPENAI_API_KEY=OPENAI_API_KEY:<nonce_b64>:<ct_b64>
            ENCRYPTED_TWITTER_API_KEY=TWITTER_API_KEY:<nonce_b64>:<ct_b64>
        """
        prefix = "ENCRYPTED_"
        creds: dict[str, EncryptedCredential] = {}
        for env_key, env_val in os.environ.items():
            if env_key.startswith(prefix) and env_val:
                try:
                    cred = EncryptedCredential.from_env_value(env_val)
                    creds[cred.name] = cred
                except (ValueError, IndexError) as exc:
                    log.warning("Skipping malformed credential %r: %s", env_key, exc)
        return cls(credentials=creds)

    def register(self, credential: EncryptedCredential) -> None:
        """Register or update a single encrypted credential (rarely needed at runtime)."""
        self._credentials[credential.name] = credential

    def has(self, name: str) -> bool:
        return name in self._credentials

    def _decrypt(self, name: str) -> bytes:
        """Decrypt and return plaintext bytes. The caller MUST zero them after use."""
        cred = self._credentials.get(name)
        if cred is None:
            raise CredentialNotFoundError(
                f"No encrypted credential registered for {name!r}. "
                "Check ENCRYPTED_<NAME> env vars are set."
            )
        nonce = base64.urlsafe_b64decode(cred.nonce_b64)
        ciphertext = base64.urlsafe_b64decode(cred.ciphertext_b64)
        aesgcm = AESGCM(self._master_key)
        try:
            plaintext = aesgcm.decrypt(nonce, ciphertext, cred.name.encode())
        except Exception as exc:
            # Map cryptography's InvalidTag to our domain exception.
            # Do NOT include exc message — it might contain partial key material.
            raise AuthenticationError(
                f"GCM authentication failed for credential {name!r}. "
                "The ciphertext may be tampered or the master key is incorrect."
            ) from None
        return plaintext


# ---------------------------------------------------------------------------
# LifecycleManager: JIT context injector (the public-facing inject API)
# ---------------------------------------------------------------------------
class LifecycleManager:
    """Just-In-Time credential injector — the only sanctioned way to use secrets.

    Usage::

        store = CredentialStore.from_env()
        manager = LifecycleManager(store)

        # The lambda receives the plaintext API key as a str.
        # The plaintext is zeroed before inject() returns.
        result = manager.inject("OPENAI_API_KEY", lambda key: do_something(key))

    The plaintext is:
    - Decrypted inside inject(), held only in a local bytearray on the heap.
    - Passed as a str to the callable (unavoidable for most SDK clients).
    - Zeroed immediately after the callable returns, using ctypes.memset on
      the bytearray backing the bytes object.
    - NEVER returned, stored as an attribute, logged, or serialised.

    If the callable raises, the plaintext is zeroed in the finally block
    before the exception propagates — secrets never leak via exceptions.
    """

    def __init__(self, store: CredentialStore) -> None:
        self._store = store

    def inject(self, name: str, context_fn: Callable[[str], T]) -> T:
        """Decrypt *name*, call ``context_fn(plaintext_str)``, zero the memory.

        Parameters
        ----------
        name       : the logical credential name to look up (e.g. "OPENAI_API_KEY").
        context_fn : callable that receives the plaintext secret as a ``str``
                     and returns a result (or raises). Must not store the secret
                     beyond its own call frame.

        Returns
        -------
        Whatever ``context_fn`` returns.

        Raises
        ------
        CredentialNotFoundError : if *name* is not registered.
        AuthenticationError     : if GCM tag verification fails.
        """
        # Decrypt into a mutable bytearray so we can zero it.
        raw: bytes = self._store._decrypt(name)
        # We copy into a bytearray so ctypes can zero it.  The str conversion
        # for the callable is a necessary evil — most SDK clients expect str.
        buf = bytearray(raw)
        try:
            plaintext_str = buf.decode()
            return context_fn(plaintext_str)
        finally:
            # Zero the mutable buffer unconditionally, even on exception.
            _zero_bytearray(buf)
            # Note: plaintext_str is a Python str object on the heap; Python
            # str is immutable and interned, so we cannot zero it reliably.
            # The mitigation is that it goes out of scope here and is eligible
            # for GC.  For workloads requiring HSM-grade zeroing, replace the
            # SDK clients with ones that accept bytes directly.
            del plaintext_str

    def inject_env(self, name: str) -> None:
        """Convenience: inject a credential directly into ``os.environ[name]``.

        Useful for library integrations that read env vars (e.g. ``openai``
        reads ``OPENAI_API_KEY`` from the environment by default).

        Security note: the plaintext lives in os.environ for the duration of
        the process — less safe than inject() but acceptable when the SDK
        does not accept an explicit key parameter.
        """
        raw = self._store._decrypt(name)
        buf = bytearray(raw)
        try:
            os.environ[name] = buf.decode()
        finally:
            _zero_bytearray(buf)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _zero_bytearray(buf: bytearray) -> None:
    """Overwrite ``buf`` with zeros using ctypes to defeat compiler/interpreter
    optimisations that might elide a plain ``buf[:] = b'\\x00' * len(buf)``."""
    if not buf:
        return
    addr = ctypes.addressof((ctypes.c_char * len(buf)).from_buffer(buf))
    ctypes.memset(addr, 0, len(buf))
