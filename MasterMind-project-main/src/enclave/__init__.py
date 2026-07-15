"""Credential Enclave: AES-256-GCM BYOK key store.

Public API
----------
    from src.enclave import CredentialStore, LifecycleManager

See enclave.py for full documentation.
"""

from .enclave import (
    AuthenticationError,
    CredentialStore,
    EncryptedCredential,
    LifecycleManager,
    encrypt_credential,
)

__all__ = [
    "AuthenticationError",
    "CredentialStore",
    "EncryptedCredential",
    "LifecycleManager",
    "encrypt_credential",
]
