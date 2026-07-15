"""Distributed Lock — Redlock-pattern idempotency guardian.

Why Redlock?
------------
A single ``SET NX EX`` on one Redis node is sufficient if your Redis
deployment is a single primary (Upstash default).  However, under a network
partition or primary failover, a ``SET NX`` that was acknowledged before the
failover may not be replicated to the new primary — a second worker could
then acquire the same lock.

The full Redlock algorithm requires N ≥ 3 independent Redis nodes.  Since
most deployments (including Upstash) use a single endpoint, this module
implements *single-node Redlock*: the same algorithm structure (random
token, TTL, Lua safe-release) but against one node.  The class accepts a
``nodes`` list so operators can add replica URLs for true multi-node Redlock
without changing call sites.

Algorithm (per lock acquisition attempt)
-----------------------------------------
  For each Redis node:
    1. Record wall-clock start time.
    2. ``SET lock:{job_id} {token} NX PX {ttl_ms}``  (atomic).
    3. Record elapsed time.
  Validity check:
    elapsed_ms < ttl_ms * drift_factor (0.01) + 2ms  AND
    successful_nodes / total_nodes > 0.5

  On failure: release any partial locks immediately.

Token format
------------
  ``{worker_id}:{uuid4}``.  The worker_id portion is human-readable for
  operator debugging; the uuid4 ensures uniqueness across restarts.

Safe release (Lua script)
--------------------------
  Atomic compare-and-delete: only delete the key if its value equals the
  token this worker set.  Prevents releasing another worker's lock if our
  own TTL expired while we were still processing.

  Script is loaded via SCRIPT LOAD / EVALSHA on first use to avoid resending
  it on every release.

Fence token / versioning
-------------------------
  Redis does not provide a monotonic sequence for lock fencing (unlike
  ZooKeeper's epoch).  In our architecture, the idempotency key in the job
  payload provides a stronger guarantee: the gateway's SET NX ensures
  exactly one job_id per idempotency_key, and the worker's duplicate-check
  (SET NX on ``idempotency:processed:{key}``) provides the final guard.
  The distributed lock's purpose is to prevent *concurrent* execution of the
  same job on two workers, not long-term replay prevention.

OOM-kill guarantee
------------------
  The lock TTL (default 330s) is intentionally longer than the max pipeline
  duration (300s).  If the worker is OOM-killed, the lock expires and the
  XAUTOCLAIM cycle re-delivers the stream message to another worker.
  XACK is called by the Worker *only after* the pipeline completes and the
  lock is about to be released — there is no window where the lock is
  released but the message is unacknowledged.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import redis

from ..logging_setup import get_logger

log = get_logger("fleet.lock")

_LOCK_PREFIX = "lock:"
_DEFAULT_TTL_MS = 330_000       # 330 seconds: longer than max pipeline (300s)
_ACQUIRE_RETRY_DELAY_S = 0.1    # wait between per-node acquire retries
_DRIFT_FACTOR = 0.01            # clock drift allowance: 1% of TTL
_DRIFT_CONSTANT_MS = 2          # fixed 2ms network overhead allowance


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class LockNotAcquiredError(Exception):
    """Raised when all acquisition attempts fail (another worker holds the lock)."""


class LockAlreadyReleasedError(Exception):
    """Raised when trying to release a lock that has already been released or expired."""


# ---------------------------------------------------------------------------
# Single-node lock context (returned by DistributedLock.acquire)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class LockHandle:
    """An acquired lock.  Use as a context manager or call release() manually.

    The handle is a value object — it does not hold a Redis connection.
    All Redis calls are dispatched back through the originating
    ``DistributedLock`` instance so the circuit-breaker logic is shared.
    """

    job_id: str
    token: str
    acquired_at: float      # monotonic timestamp
    ttl_ms: int
    _lock: "DistributedLock" = field(repr=False, compare=False)
    _released: bool = field(default=False, init=False)

    def release(self) -> None:
        """Safe release via Lua compare-and-delete.  Idempotent."""
        if not self._released:
            self._lock._release(self)
            self._released = True

    def is_valid(self) -> bool:
        """Return True if the lock has not yet expired locally.

        This is a conservative local check; the authoritative check is on the
        Redis node.  Use this to decide whether to extend the lock TTL before
        a long operation, not as a definitive validity guarantee.
        """
        elapsed_ms = (time.monotonic() - self.acquired_at) * 1000
        return elapsed_ms < self.ttl_ms * (1 - _DRIFT_FACTOR)

    def __enter__(self) -> "LockHandle":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class LockConfig:
    redis_urls: list[str] = field(
        default_factory=lambda: [
            os.environ.get("UPSTASH_REDIS_REST_URL", "redis://localhost:6379")
        ]
    )
    ttl_ms: int = _DEFAULT_TTL_MS
    retry_count: int = 3        # acquisition attempts before giving up
    retry_delay_s: float = _ACQUIRE_RETRY_DELAY_S


# ---------------------------------------------------------------------------
# Lua safe-release script
# ---------------------------------------------------------------------------
# Language: Redis Lua.  Runs atomically inside the Redis server.
# KEYS[1] = lock key, ARGV[1] = expected token.
_LUA_RELEASE = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


# ---------------------------------------------------------------------------
# DistributedLock
# ---------------------------------------------------------------------------
class DistributedLock:
    """Redlock-pattern distributed lock over one or more Redis nodes.

    Construct once per Worker instance (shared across jobs).
    Thread-safe: each ``acquire()`` call uses its own token and operates
    independently on the Redis nodes.

    Usage::

        lock = DistributedLock(worker_id="w-abc123")

        with lock.acquire("job_id_xyz") as handle:
            # exclusive section — safe to execute pipeline
            pipeline.run()
        # handle.release() called automatically on block exit

    Raises ``LockNotAcquiredError`` if the lock cannot be acquired after
    ``retry_count`` attempts (another worker holds it).
    """

    def __init__(
        self,
        worker_id: str,
        config: LockConfig | None = None,
    ) -> None:
        self._worker_id = worker_id
        self._cfg = config or LockConfig()
        self._clients: list[redis.Redis] = [
            redis.from_url(
                url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=3,
            )
            for url in self._cfg.redis_urls
        ]
        # Cache EVALSHA per client to avoid resending the script.
        self._script_shas: dict[int, str | None] = {i: None for i in range(len(self._clients))}

    # -- Public API ----------------------------------------------------------

    @contextmanager
    def acquire(self, job_id: str) -> Iterator[LockHandle]:
        """Acquire the lock as a context manager.

        Yields a ``LockHandle`` if successful.
        Raises ``LockNotAcquiredError`` if all retries are exhausted.
        """
        handle = self._try_acquire(job_id)
        try:
            yield handle
        finally:
            handle.release()

    def try_acquire(self, job_id: str) -> LockHandle | None:
        """Non-raising acquisition attempt.  Returns None if not acquired."""
        try:
            return self._try_acquire(job_id)
        except LockNotAcquiredError:
            return None

    # -- Acquisition ---------------------------------------------------------

    def _try_acquire(self, job_id: str) -> LockHandle:
        """Attempt acquisition up to retry_count times."""
        for attempt in range(1, self._cfg.retry_count + 1):
            handle = self._acquire_once(job_id)
            if handle is not None:
                log.debug("Lock acquired: job_id=%s attempt=%d", job_id, attempt)
                return handle
            if attempt < self._cfg.retry_count:
                log.debug(
                    "Lock not acquired for job_id=%s (attempt %d/%d) — retrying in %.2fs.",
                    job_id, attempt, self._cfg.retry_count, self._cfg.retry_delay_s,
                )
                time.sleep(self._cfg.retry_delay_s)

        raise LockNotAcquiredError(
            f"Could not acquire lock for job_id={job_id!r} after "
            f"{self._cfg.retry_count} attempts."
        )

    def _acquire_once(self, job_id: str) -> LockHandle | None:
        """Single Redlock acquisition round across all nodes."""
        token = f"{self._worker_id}:{uuid.uuid4().hex}"
        key = f"{_LOCK_PREFIX}{job_id}"
        start_ms = time.monotonic() * 1000

        acquired_count = 0
        acquired_on: list[int] = []

        for idx, client in enumerate(self._clients):
            try:
                result = client.set(key, token, nx=True, px=self._cfg.ttl_ms)
                if result is not None:
                    acquired_count += 1
                    acquired_on.append(idx)
            except redis.RedisError as exc:
                log.warning("Lock SET failed on node %d: %s", idx, exc)

        elapsed_ms = time.monotonic() * 1000 - start_ms
        validity_ms = self._cfg.ttl_ms - elapsed_ms - (self._cfg.ttl_ms * _DRIFT_FACTOR + _DRIFT_CONSTANT_MS)

        quorum = len(self._clients) // 2 + 1
        if acquired_count >= quorum and validity_ms > 0:
            return LockHandle(
                job_id=job_id,
                token=token,
                acquired_at=time.monotonic(),
                ttl_ms=int(validity_ms),
                _lock=self,
            )

        # Quorum not reached — release any partial acquisitions immediately.
        self._release_on_nodes(key, token, acquired_on)
        return None

    # -- Release -------------------------------------------------------------

    def _release(self, handle: LockHandle) -> None:
        """Release the lock on all nodes."""
        key = f"{_LOCK_PREFIX}{handle.job_id}"
        self._release_on_nodes(key, handle.token, list(range(len(self._clients))))

    def _release_on_nodes(self, key: str, token: str, node_indices: list[int]) -> None:
        """Lua safe-release on the specified nodes.  Best-effort; never raises."""
        for idx in node_indices:
            client = self._clients[idx]
            try:
                sha = self._ensure_script(idx, client)
                if sha:
                    result = client.evalsha(sha, 1, key, token)
                else:
                    # Fallback if SCRIPT LOAD is not available (very old Redis).
                    result = client.eval(_LUA_RELEASE, 1, key, token)
                if not result:
                    log.warning(
                        "Lock release on node %d returned 0 for key %s "
                        "(token mismatch or TTL expired — not an error if processing took too long).",
                        idx, key,
                    )
                else:
                    log.debug("Lock released on node %d: key=%s", idx, key)
            except redis.RedisError as exc:
                # Non-fatal: the TTL will expire the lock automatically.
                log.warning("Lock release failed on node %d: %s", idx, exc)

    def _ensure_script(self, idx: int, client: redis.Redis) -> str | None:
        """SCRIPT LOAD the Lua release script once per client and cache the SHA."""
        if self._script_shas[idx] is None:
            try:
                self._script_shas[idx] = client.script_load(_LUA_RELEASE)
            except redis.RedisError as exc:
                log.warning("SCRIPT LOAD failed on node %d: %s — will use EVAL.", idx, exc)
                self._script_shas[idx] = None  # will retry next time
        return self._script_shas[idx]
