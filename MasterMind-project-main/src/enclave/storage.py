"""storage.py — asyncpg-backed persistence for the Credential Enclave.

Layering
--------
    crypto.py    pure AES-256-GCM primitives (no I/O)
    storage.py   THIS FILE: DB persistence of SealedBlobs + JIT injection
    enclave.py   env-var-based store (single-tenant / CI mode)

The database stores ONLY sealed material (see migrations/001_credential_enclave.sql).
The master key never leaves the process env; a full database dump is
cryptographically useless without it.

Failure model (extreme error handling)
--------------------------------------
Every public method distinguishes three failure classes:

  StorageUnavailableError   — the DB is unreachable / pool exhausted / timeout.
                              Callers should return 503 and let the fleet's
                              retry engine re-attempt. NEVER crashes the worker.
  CredentialMissingError    — the (user, name) pair has no row. A 404-class
                              user error; retrying will not help.
  crypto.DecryptionError    — tampered row or wrong master key. This is a
                              SECURITY EVENT: it is audit-logged (metadata only)
                              and re-raised. Retrying will not help.

Connection strategy
-------------------
A lazily created asyncpg pool (min 1 / max 5 — serverless-friendly) with
per-operation timeouts. If pool creation itself fails, the error is wrapped
in StorageUnavailableError and the pool attribute stays None so the next
call retries a fresh connection rather than reusing a poisoned object.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

import asyncpg

from ..logging_setup import get_logger
from . import crypto
from .crypto import DecryptionError, SealedBlob, SecretBuffer

log = get_logger("enclave.storage")

T = TypeVar("T")

_MIGRATION_FILE = Path(__file__).resolve().parents[2] / "migrations" / "001_credential_enclave.sql"
_OP_TIMEOUT_S = 5.0          # per-query timeout
_CONNECT_TIMEOUT_S = 4.0     # pool acquisition timeout


class StorageUnavailableError(Exception):
    """The database is unreachable. Transient — safe to retry with backoff."""


class CredentialMissingError(KeyError):
    """No sealed credential exists for this (user, name). Not retryable."""


class CredentialRepository:
    """Async repository for sealed credentials with JIT injection.

    Construct once per process (worker startup); share across tasks.
    All methods are coroutine-safe.
    """

    def __init__(self, *, dsn: str | None = None, master_key: bytes | None = None) -> None:
        self._dsn = dsn or os.environ.get("DATABASE_URL", "")
        if not self._dsn:
            raise RuntimeError("DATABASE_URL is not set; the Credential Enclave requires Postgres.")
        self._master_key = master_key or crypto.load_master_key(os.environ["ENCLAVE_MASTER_KEY"])
        self._pool: asyncpg.Pool | None = None
        self._pool_lock = asyncio.Lock()

    # -- Pool lifecycle -------------------------------------------------------

    async def _get_pool(self) -> asyncpg.Pool:
        """Lazily create the pool; on failure leave state clean for retry."""
        if self._pool is not None:
            return self._pool
        async with self._pool_lock:
            if self._pool is not None:  # double-checked: another task won the race
                return self._pool
            try:
                self._pool = await asyncpg.create_pool(
                    dsn=self._dsn,
                    min_size=1,
                    max_size=5,
                    command_timeout=_OP_TIMEOUT_S,
                    timeout=_CONNECT_TIMEOUT_S,
                )
            except (OSError, asyncpg.PostgresError, asyncio.TimeoutError) as exc:
                self._pool = None  # do not cache a poisoned pool
                raise StorageUnavailableError(
                    "Credential database is unreachable."
                ) from exc
            return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def migrate(self) -> None:
        """Apply the idempotent schema migration. Call once at worker startup."""
        sql = _MIGRATION_FILE.read_text()
        await self._execute(lambda conn: conn.execute(sql))

    # -- Guarded execution helper ----------------------------------------------

    async def _execute(self, op: Callable[[asyncpg.Connection], Awaitable[T]]) -> T:
        """Run *op* on a pooled connection, mapping infra errors to StorageUnavailableError."""
        pool = await self._get_pool()
        try:
            async with pool.acquire(timeout=_CONNECT_TIMEOUT_S) as conn:
                return await op(conn)
        except (asyncpg.PostgresConnectionError, asyncpg.InterfaceError,
                OSError, asyncio.TimeoutError) as exc:
            # Connection-class failures: drop the pool so the next call rebuilds it.
            self._pool = None
            raise StorageUnavailableError("Credential database is unreachable.") from exc
        # NOTE: asyncpg.PostgresError subclasses that are NOT connection errors
        # (e.g. constraint violations) propagate untouched — they are bugs or
        # user errors, not transient infrastructure failures.

    # -- Write path ------------------------------------------------------------

    async def store_credential(self, external_user_id: str, name: str, plaintext: str) -> None:
        """Seal and upsert a credential for a user. Plaintext is zeroed before return."""
        buf = bytearray(plaintext.encode())
        try:
            blob = crypto.seal(self._master_key, name, bytes(buf))
        finally:
            crypto.zero_buffer(buf)

        async def op(conn: asyncpg.Connection) -> None:
            async with conn.transaction():
                user_id = await conn.fetchval(
                    """
                    INSERT INTO enclave_users (external_id) VALUES ($1)
                    ON CONFLICT (external_id) DO UPDATE SET external_id = EXCLUDED.external_id
                    RETURNING id
                    """,
                    external_user_id,
                )
                await conn.execute(
                    """
                    INSERT INTO enclave_credentials (user_id, name, nonce, ciphertext)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (user_id, name) DO UPDATE
                        SET nonce = EXCLUDED.nonce,
                            ciphertext = EXCLUDED.ciphertext,
                            updated_at = now()
                    """,
                    user_id, name, blob.nonce, blob.ciphertext,
                )
                await conn.execute(
                    "INSERT INTO enclave_audit_log (user_id, action, credential) VALUES ($1, 'store', $2)",
                    user_id, name,
                )

        await self._execute(op)
        log.info("Stored sealed credential %s for user %s", name, external_user_id)

    async def delete_credential(self, external_user_id: str, name: str) -> bool:
        """Delete a credential. Returns True if a row was removed."""

        async def op(conn: asyncpg.Connection) -> bool:
            result = await conn.execute(
                """
                DELETE FROM enclave_credentials c
                USING enclave_users u
                WHERE c.user_id = u.id AND u.external_id = $1 AND c.name = $2
                """,
                external_user_id, name,
            )
            return result.endswith("1")

        return await self._execute(op)

    # -- Read / JIT injection path ----------------------------------------------

    async def _fetch_blob(self, external_user_id: str, name: str) -> SealedBlob:
        async def op(conn: asyncpg.Connection) -> asyncpg.Record | None:
            return await conn.fetchrow(
                """
                SELECT c.nonce, c.ciphertext
                FROM enclave_credentials c
                JOIN enclave_users u ON u.id = c.user_id
                WHERE u.external_id = $1 AND c.name = $2 AND u.disabled_at IS NULL
                """,
                external_user_id, name,
            )

        row = await self._execute(op)
        if row is None:
            raise CredentialMissingError(
                f"No credential {name!r} stored for user {external_user_id!r}."
            )
        return SealedBlob(aad=name, nonce=bytes(row["nonce"]), ciphertext=bytes(row["ciphertext"]))

    async def inject(
        self,
        external_user_id: str,
        name: str,
        context_fn: Callable[[str], Awaitable[T] | T],
    ) -> T:
        """JIT injection: decrypt → call → zero, exactly like LifecycleManager.inject.

        The plaintext exists only for the duration of *context_fn*'s call frame
        and is zeroed in a finally block even if the callable raises. Supports
        both sync and async callables.

        Raises
        ------
        StorageUnavailableError — DB down (retryable, return 503 upstream).
        CredentialMissingError  — no such credential (404-class, not retryable).
        DecryptionError         — tamper / wrong key (security event, audited).
        """
        blob = await self._fetch_blob(external_user_id, name)
        try:
            plaintext_buf = crypto.unseal(self._master_key, blob)
        except DecryptionError:
            # Security event: audit with metadata only, best-effort (never mask
            # the original error if the audit write itself fails).
            try:
                await self._audit(external_user_id, "decrypt_failure", name)
            except StorageUnavailableError:
                log.error("Audit write failed while recording decrypt_failure for %s", name)
            raise

        try:
            with SecretBuffer(plaintext_buf) as secret:
                result = context_fn(secret.decode())
                if asyncio.iscoroutine(result):
                    result = await result
        finally:
            # SecretBuffer zeroed plaintext_buf's copy; zero the original too.
            crypto.zero_buffer(plaintext_buf)

        # Fire-and-forget usage bookkeeping — failures here must NOT fail the job.
        try:
            await self._touch_usage(external_user_id, name)
        except StorageUnavailableError:
            log.warning("Usage bookkeeping skipped for %s (DB unavailable)", name)
        return result  # type: ignore[return-value]

    # -- Key rotation -------------------------------------------------------------

    async def rotate_master_key(self, new_key: bytes, *, new_version: int) -> int:
        """Re-seal every credential under *new_key*. Returns rows rotated.

        Runs in batches inside a transaction per row so a mid-rotation crash
        leaves the table in a mixed-but-valid state (each row's key_version
        records which key sealed it).
        """
        rotated = 0

        async def op(conn: asyncpg.Connection) -> list[asyncpg.Record]:
            return await conn.fetch(
                "SELECT id, name, nonce, ciphertext FROM enclave_credentials WHERE key_version < $1",
                new_version,
            )

        rows = await self._execute(op)
        for row in rows:
            blob = SealedBlob(aad=row["name"], nonce=bytes(row["nonce"]), ciphertext=bytes(row["ciphertext"]))
            new_blob = crypto.rewrap(self._master_key, new_key, blob)

            async def update(conn: asyncpg.Connection, _id=row["id"], _b=new_blob) -> None:
                await conn.execute(
                    """
                    UPDATE enclave_credentials
                    SET nonce = $2, ciphertext = $3, key_version = $4, updated_at = now()
                    WHERE id = $1
                    """,
                    _id, _b.nonce, _b.ciphertext, new_version,
                )

            await self._execute(update)
            rotated += 1

        self._master_key = new_key
        log.info("Rotated %d credentials to key version %d", rotated, new_version)
        return rotated

    # -- Internals -----------------------------------------------------------------

    async def _touch_usage(self, external_user_id: str, name: str) -> None:
        async def op(conn: asyncpg.Connection) -> None:
            await conn.execute(
                """
                UPDATE enclave_credentials c
                SET last_used_at = now(), use_count = c.use_count + 1
                FROM enclave_users u
                WHERE c.user_id = u.id AND u.external_id = $1 AND c.name = $2
                """,
                external_user_id, name,
            )

        await self._execute(op)

    async def _audit(self, external_user_id: str, action: str, credential: str) -> None:
        async def op(conn: asyncpg.Connection) -> None:
            await conn.execute(
                """
                INSERT INTO enclave_audit_log (user_id, action, credential)
                SELECT u.id, $2, $3 FROM enclave_users u WHERE u.external_id = $1
                """,
                external_user_id, action, credential,
            )

        await self._execute(op)
