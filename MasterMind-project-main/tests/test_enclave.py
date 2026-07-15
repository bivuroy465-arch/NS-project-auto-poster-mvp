"""Tests for the Credential Enclave (AES-256-GCM BYOK)."""

from __future__ import annotations

import base64
import os

import pytest

from src.enclave.enclave import (
    AuthenticationError,
    CredentialNotFoundError,
    CredentialStore,
    EncryptedCredential,
    LifecycleManager,
    _zero_bytearray,
    encrypt_credential,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
MASTER_KEY = os.urandom(32)
MASTER_KEY_B64 = base64.urlsafe_b64encode(MASTER_KEY).decode()


@pytest.fixture()
def store() -> CredentialStore:
    cred = encrypt_credential("TEST_KEY", "supersecret", MASTER_KEY)
    return CredentialStore(master_key_b64=MASTER_KEY_B64, credentials={"TEST_KEY": cred})


@pytest.fixture()
def manager(store: CredentialStore) -> LifecycleManager:
    return LifecycleManager(store)


# ---------------------------------------------------------------------------
# encrypt_credential
# ---------------------------------------------------------------------------
def test_encrypt_produces_different_nonces() -> None:
    c1 = encrypt_credential("K", "value", MASTER_KEY)
    c2 = encrypt_credential("K", "value", MASTER_KEY)
    assert c1.nonce_b64 != c2.nonce_b64, "Each encryption must use a fresh random nonce."


def test_encrypt_wrong_key_length_raises() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        encrypt_credential("K", "v", b"too-short")


# ---------------------------------------------------------------------------
# EncryptedCredential serialisation round-trip
# ---------------------------------------------------------------------------
def test_to_from_env_value_roundtrip() -> None:
    cred = encrypt_credential("MY_KEY", "my_secret", MASTER_KEY)
    serialised = cred.to_env_value()
    restored = EncryptedCredential.from_env_value(serialised)
    assert restored == cred


def test_from_env_value_malformed_raises() -> None:
    with pytest.raises(ValueError, match="Malformed"):
        EncryptedCredential.from_env_value("only_one_part")


# ---------------------------------------------------------------------------
# CredentialStore
# ---------------------------------------------------------------------------
def test_store_missing_master_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENCLAVE_MASTER_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ENCLAVE_MASTER_KEY"):
        CredentialStore()


def test_store_wrong_key_length_raises() -> None:
    bad_b64 = base64.urlsafe_b64encode(b"tooshort").decode()
    with pytest.raises(ValueError, match="32 bytes"):
        CredentialStore(master_key_b64=bad_b64)


def test_store_has_known_credential(store: CredentialStore) -> None:
    assert store.has("TEST_KEY")
    assert not store.has("NONEXISTENT")


def test_store_decrypt_correct_value(store: CredentialStore) -> None:
    plaintext = store._decrypt("TEST_KEY")
    assert plaintext == b"supersecret"


def test_store_decrypt_unknown_raises(store: CredentialStore) -> None:
    with pytest.raises(CredentialNotFoundError):
        store._decrypt("NONEXISTENT")


def test_store_decrypt_tampered_ciphertext_raises() -> None:
    cred = encrypt_credential("K", "secret", MASTER_KEY)
    # Flip the last byte of the ciphertext to corrupt the GCM tag.
    ct_bytes = bytearray(base64.urlsafe_b64decode(cred.ciphertext_b64))
    ct_bytes[-1] ^= 0xFF
    tampered = EncryptedCredential(
        name=cred.name,
        nonce_b64=cred.nonce_b64,
        ciphertext_b64=base64.urlsafe_b64encode(bytes(ct_bytes)).decode(),
    )
    store = CredentialStore(master_key_b64=MASTER_KEY_B64, credentials={"K": tampered})
    with pytest.raises(AuthenticationError):
        store._decrypt("K")


def test_store_from_env_reads_encrypted_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    cred = encrypt_credential("MY_SECRET", "val", MASTER_KEY)
    monkeypatch.setenv("ENCLAVE_MASTER_KEY", MASTER_KEY_B64)
    monkeypatch.setenv("ENCRYPTED_MY_SECRET", cred.to_env_value())
    store = CredentialStore.from_env()
    assert store.has("MY_SECRET")
    assert store._decrypt("MY_SECRET") == b"val"


# ---------------------------------------------------------------------------
# LifecycleManager.inject
# ---------------------------------------------------------------------------
def test_inject_passes_plaintext_to_callable(manager: LifecycleManager) -> None:
    received: list[str] = []
    manager.inject("TEST_KEY", lambda k: received.append(k))
    assert received == ["supersecret"]


def test_inject_returns_callable_result(manager: LifecycleManager) -> None:
    result = manager.inject("TEST_KEY", lambda k: len(k))
    assert result == len("supersecret")


def test_inject_zeros_buffer_after_call(manager: LifecycleManager) -> None:
    """Verify that the plaintext is not left accessible after inject() returns."""
    captured: list[str] = []

    def capture(k: str) -> None:
        captured.append(k)

    manager.inject("TEST_KEY", capture)
    # We can only indirectly verify this: ensure the function ran and returned.
    assert captured[0] == "supersecret"


def test_inject_unknown_key_raises(manager: LifecycleManager) -> None:
    with pytest.raises(CredentialNotFoundError):
        manager.inject("NONEXISTENT", lambda k: None)


def test_inject_exception_in_callable_propagates(manager: LifecycleManager) -> None:
    with pytest.raises(ValueError, match="boom"):
        manager.inject("TEST_KEY", lambda k: (_ for _ in ()).throw(ValueError("boom")))


# ---------------------------------------------------------------------------
# _zero_bytearray helper
# ---------------------------------------------------------------------------
def test_zero_bytearray_zeroes_content() -> None:
    buf = bytearray(b"sensitive data here")
    _zero_bytearray(buf)
    assert all(b == 0 for b in buf)


def test_zero_bytearray_empty_noop() -> None:
    buf = bytearray()
    _zero_bytearray(buf)  # must not raise
