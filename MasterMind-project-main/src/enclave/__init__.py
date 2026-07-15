"""Credential Enclave: AES-256-GCM BYOK key store.

Public API
----------
    from src.enclave import CredentialStore, LifecycleManager

See enclave.py for full documentation.
"""

from .crypto import DecryptionError, SealedBlob, SecretBuffer, seal, unseal, zero_buffer
from .enclave import (
    AuthenticationError,
    CredentialStore,
    EncryptedCredential,
    LifecycleManager,
    encrypt_credential,
)
from .storage import CredentialMissingError, CredentialRepository, StorageUnavailableError

__all__ = [
    "AuthenticationError",
    "CredentialMissingError",
    "CredentialRepository",
    "CredentialStore",
    "DecryptionError",
    "EncryptedCredential",
    "LifecycleManager",
    "SealedBlob",
    "SecretBuffer",
    "StorageUnavailableError",
    "encrypt_credential",
    "seal",
    "unseal",
    "zero_buffer",
]
