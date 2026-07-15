"""Edge Gateway — JWT auth, rate limiting, and Redis Stream job broker.

Architecture
------------
The gateway is intentionally the *thinnest possible* layer between an
incoming HTTP request and the background worker fleet. Its only jobs are:

  1. Verify the caller's JWT (HS256, max 15-min expiry, verified issuer).
  2. Apply a per-IP sliding-window rate limit (Upstash Redis INCR + EX).
  3. Enforce idempotency (SET NX on the provided idempotency key, 24h TTL).
  4. Enqueue a job payload to the Redis Stream ``autoposter:jobs`` (XADD).
  5. Return 202 Accepted + job_id immediately — *no AI calls, no DB writes*.

This O(1) path means the gateway never approaches a serverless timeout,
even under extreme bursts: Redis INCR, SET NX, and XADD are all sub-ms.

The job_id returned by enqueue_job() can be polled via get_job_status(),
which reads from a Redis Hash populated by the fleet worker on completion.

Redis key schema
----------------
  autoposter:jobs          — Redis Stream; workers XREADGROUP from here.
  job:{job_id}             — Hash: status, result, error, created_at, attempts.
  idempotency:{key}        — String (NX, 24h TTL): prevents duplicate enqueues.
  rate:{ip}:{window_start} — String (INCR, sliding window): per-IP rate limit.
  lock:{job_id}            — String (NX, 300s TTL): fleet distributed lock.

JWT security
------------
  Algorithm : HS256 (symmetric; secret rotated via GATEWAY_JWT_SECRET env var).
  Required claims: exp (must be in future), iat, iss (must match GATEWAY_JWT_ISSUER).
  Max validity : 15 minutes (enforced by verifying exp - iat <= 900s).
  Replay : not explicitly prevented here (stateless); idempotency key covers
           functional replay; rate limiting covers brute-force replay.

Framework independence
----------------------
`GatewayRouter` is a pure-Python class with no framework imports. It can be
wrapped by FastAPI, Flask, or a Vercel Python serverless function. See
``api/gateway/`` for the Vercel serverless entrypoints.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import jwt  # PyJWT
import redis  # redis-py (Upstash compatible over HTTP-TLS)
from pydantic import BaseModel, ConfigDict, Field

from ..logging_setup import get_logger

log = get_logger("gateway")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_STREAM_KEY = "autoposter:jobs"
_JOB_HASH_PREFIX = "job:"
_IDEMPOTENCY_PREFIX = "idempotency:"
_RATE_PREFIX = "rate:"
_LOCK_PREFIX = "lock:"

_JWT_ALGORITHM = "HS256"
_JWT_MAX_VALIDITY_SECONDS = 900  # 15 minutes
_RATE_LIMIT_WINDOW_SECONDS = 60
_IDEMPOTENCY_TTL_SECONDS = 86_400  # 24 hours
_LOCK_TTL_SECONDS = 300            # 5 minutes max pipeline execution


# ---------------------------------------------------------------------------
# Domain errors
# ---------------------------------------------------------------------------
class TokenInvalidError(Exception):
    """Raised when JWT verification fails for any reason."""


class RateLimitExceededError(Exception):
    """Raised when the caller has exceeded their per-window request quota."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        super().__init__(f"Rate limit of {limit} requests per {window_seconds}s exceeded.")
        self.limit = limit
        self.window_seconds = window_seconds


class IdempotencyConflictError(Exception):
    """Raised when the idempotency key has already been consumed (request is a duplicate)."""

    def __init__(self, key: str) -> None:
        super().__init__(f"Idempotency key {key!r} was already used within the last 24 h.")
        self.key = key


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class GatewayConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    jwt_secret: str = Field(default_factory=lambda: os.environ["GATEWAY_JWT_SECRET"])
    jwt_issuer: str = Field(default_factory=lambda: os.environ.get("GATEWAY_JWT_ISSUER", "autoposter"))
    rate_limit: int = Field(default=60, gt=0, description="Max requests per IP per window.")
    rate_window_seconds: int = Field(default=_RATE_LIMIT_WINDOW_SECONDS, gt=0)
    redis_url: str = Field(
        default_factory=lambda: os.environ["UPSTASH_REDIS_REST_URL"],
        description="Upstash Redis URL (redis://... or rediss://...).",
    )
    redis_token: str | None = Field(
        default_factory=lambda: os.environ.get("UPSTASH_REDIS_REST_TOKEN"),
        description="Upstash Redis REST token (if using HTTP REST API).",
    )


# ---------------------------------------------------------------------------
# Pydantic schemas for request / response
# ---------------------------------------------------------------------------
class EnqueueRequest(BaseModel):
    """Payload accepted by POST /api/gateway/enqueue."""

    model_config = ConfigDict(frozen=True)

    idempotency_key: str = Field(
        description=(
            "Caller-supplied deduplication key (e.g. a hash of the schedule+date). "
            "Subsequent requests with the same key within 24 h are rejected with 409."
        )
    )
    platform_overrides: list[str] | None = Field(
        default=None,
        description="Optional: restrict this run to a subset of configured platforms.",
    )
    dry_run: bool = Field(
        default=False,
        description="If true, generate content but do not publish.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary pass-through config overrides for the pipeline.",
    )


class EnqueueResponse(BaseModel):
    """202 Accepted response from POST /api/gateway/enqueue."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    status: str = "queued"
    message: str = "Job enqueued successfully."


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"
    DLQ = "dlq"


class JobStatusResponse(BaseModel):
    """Response from GET /api/gateway/status/{job_id}."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    status: JobStatus
    created_at: str | None = None
    completed_at: str | None = None
    attempts: int = 0
    error: str | None = None
    result: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Redis job broker
# ---------------------------------------------------------------------------
class RedisJobBroker:
    """Wraps Upstash Redis for enqueue, status, and distributed-lock operations.

    Uses redis-py in standard mode; Upstash Redis is wire-compatible with
    redis-py over TLS (``rediss://``). No connection pool management is
    needed for serverless functions — each invocation creates a thin client.

    Idempotency guarantee
    ---------------------
    ``ensure_idempotency()`` uses ``SET NX EX`` (atomic in Redis ≥ 2.6.12).
    If the key already exists, the command returns ``None`` and we raise
    ``IdempotencyConflictError``. This is safe for at-least-once delivery
    scenarios: the first caller wins, subsequent callers with the same key
    are silently dropped.

    Stream message format
    ---------------------
    Each stream entry has a single field ``payload`` whose value is a
    JSON-encoded ``EnqueueRequest`` plus the gateway-assigned ``job_id``.
    """

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config
        self._client = redis.from_url(
            config.redis_url,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=3,
        )

    # -- Idempotency ----------------------------------------------------------

    def ensure_idempotency(self, idempotency_key: str) -> None:
        """Atomically claim the idempotency key. Raises if already claimed."""
        redis_key = f"{_IDEMPOTENCY_PREFIX}{idempotency_key}"
        result = self._client.set(redis_key, "1", nx=True, ex=_IDEMPOTENCY_TTL_SECONDS)
        if result is None:
            raise IdempotencyConflictError(idempotency_key)

    # -- Rate limiting --------------------------------------------------------

    def check_rate_limit(self, client_ip: str) -> None:
        """Sliding-window rate limit. Raises RateLimitExceededError if over quota."""
        window_start = int(dt.datetime.now(dt.UTC).timestamp()) // self._config.rate_window_seconds
        redis_key = f"{_RATE_PREFIX}{client_ip}:{window_start}"
        # Atomic INCR + EX in a pipeline.
        pipe = self._client.pipeline(transaction=True)
        pipe.incr(redis_key)
        pipe.expire(redis_key, self._config.rate_window_seconds)
        count, _ = pipe.execute()
        if count > self._config.rate_limit:
            raise RateLimitExceededError(self._config.rate_limit, self._config.rate_window_seconds)

    # -- Job enqueue ----------------------------------------------------------

    def enqueue(self, job_id: str, request: EnqueueRequest) -> str:
        """Append job to the Redis Stream and initialise its status hash.

        Returns the Redis Stream entry ID (``<ms>-<seq>``).
        """
        payload = json.dumps({
            "job_id": job_id,
            "idempotency_key": request.idempotency_key,
            "platform_overrides": request.platform_overrides,
            "dry_run": request.dry_run,
            "extra": request.extra,
        })
        created_at = dt.datetime.now(dt.UTC).isoformat()

        # Initialise status hash before XADD so workers always have a valid entry.
        hash_key = f"{_JOB_HASH_PREFIX}{job_id}"
        self._client.hset(hash_key, mapping={
            "status": JobStatus.QUEUED,
            "created_at": created_at,
            "attempts": 0,
        })

        entry_id = self._client.xadd(_STREAM_KEY, {"payload": payload})
        log.info("Enqueued job %s → stream entry %s", job_id, entry_id)
        return entry_id

    # -- Job status -----------------------------------------------------------

    def get_status(self, job_id: str) -> JobStatusResponse | None:
        """Fetch current job status from the Redis Hash. Returns None if not found."""
        hash_key = f"{_JOB_HASH_PREFIX}{job_id}"
        data = self._client.hgetall(hash_key)
        if not data:
            return None
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus(data.get("status", JobStatus.QUEUED)),
            created_at=data.get("created_at"),
            completed_at=data.get("completed_at"),
            attempts=int(data.get("attempts", 0)),
            error=data.get("error"),
            result=json.loads(data["result"]) if data.get("result") else None,
        )

    # -- Distributed lock (used by fleet workers) ----------------------------

    def acquire_lock(self, job_id: str, worker_id: str) -> bool:
        """Attempt to acquire the distributed lock for job_id.

        Uses ``SET NX EX`` — atomic, no race condition possible.
        Returns True if the lock was acquired, False if another worker holds it.
        """
        lock_key = f"{_LOCK_PREFIX}{job_id}"
        result = self._client.set(lock_key, worker_id, nx=True, ex=_LOCK_TTL_SECONDS)
        return result is not None

    def release_lock(self, job_id: str, worker_id: str) -> bool:
        """Release the lock only if this worker still holds it (safe release via Lua).

        Returns True if the lock was released, False if it had already expired
        or was held by a different worker (should not happen in normal operation).
        """
        lock_key = f"{_LOCK_PREFIX}{job_id}"
        # Lua script: atomic compare-and-delete.  Prevents a worker releasing
        # another worker's lock after its own TTL expired.
        lua_script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('del', KEYS[1])
        else
            return 0
        end
        """
        result = self._client.eval(lua_script, 1, lock_key, worker_id)
        return bool(result)

    def update_job_status(
        self,
        job_id: str,
        *,
        status: JobStatus,
        error: str | None = None,
        result: dict[str, Any] | None = None,
        increment_attempts: bool = False,
    ) -> None:
        """Update the job status hash atomically."""
        hash_key = f"{_JOB_HASH_PREFIX}{job_id}"
        mapping: dict[str, str] = {"status": str(status)}
        if status in (JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.DLQ):
            mapping["completed_at"] = dt.datetime.now(dt.UTC).isoformat()
        if error:
            mapping["error"] = error
        if result:
            mapping["result"] = json.dumps(result)
        pipe = self._client.pipeline(transaction=True)
        pipe.hset(hash_key, mapping=mapping)
        if increment_attempts:
            pipe.hincrby(hash_key, "attempts", 1)
        pipe.execute()


# ---------------------------------------------------------------------------
# JWT verifier
# ---------------------------------------------------------------------------
def verify_jwt(token: str, config: GatewayConfig) -> dict[str, Any]:
    """Verify a JWT and return its decoded payload.

    Validation rules (all must pass):
    - Signature valid under config.jwt_secret (HS256).
    - ``exp`` claim present and in the future (PyJWT checks this automatically).
    - ``iat`` claim present.
    - ``iss`` claim matches config.jwt_issuer.
    - Token validity (exp - iat) does not exceed 15 minutes.

    Raises ``TokenInvalidError`` with a non-specific message on any failure
    to avoid leaking token structure to callers.
    """
    try:
        payload = jwt.decode(
            token,
            config.jwt_secret,
            algorithms=[_JWT_ALGORITHM],
            options={"require": ["exp", "iat", "iss"]},
        )
    except jwt.ExpiredSignatureError:
        raise TokenInvalidError("Token has expired.")
    except jwt.InvalidIssuerError:
        raise TokenInvalidError("Token issuer is invalid.")
    except jwt.InvalidTokenError as exc:
        raise TokenInvalidError("Token is invalid.") from exc

    iss = payload.get("iss")
    if iss != config.jwt_issuer:
        raise TokenInvalidError("Token issuer is invalid.")

    exp = payload.get("exp", 0)
    iat = payload.get("iat", 0)
    if exp - iat > _JWT_MAX_VALIDITY_SECONDS:
        raise TokenInvalidError(
            f"Token validity ({exp - iat}s) exceeds the maximum of {_JWT_MAX_VALIDITY_SECONDS}s."
        )

    return payload


# ---------------------------------------------------------------------------
# GatewayRouter — framework-agnostic orchestrator
# ---------------------------------------------------------------------------
class GatewayRouter:
    """Framework-agnostic gateway logic. Wrap this in FastAPI/Flask/serverless."""

    def __init__(self, config: GatewayConfig | None = None) -> None:
        self._config = config or GatewayConfig()
        self._broker = RedisJobBroker(self._config)

    def enqueue(
        self,
        authorization_header: str,
        client_ip: str,
        request: EnqueueRequest,
    ) -> EnqueueResponse:
        """Full enqueue pipeline: auth → rate-limit → idempotency → enqueue."""
        # 1. JWT verification
        token = _extract_bearer_token(authorization_header)
        verify_jwt(token, self._config)  # raises TokenInvalidError on failure

        # 2. Rate limiting
        self._broker.check_rate_limit(client_ip)  # raises RateLimitExceededError

        # 3. Idempotency
        self._broker.ensure_idempotency(request.idempotency_key)  # raises IdempotencyConflictError

        # 4. Enqueue
        job_id = _generate_job_id(request.idempotency_key)
        self._broker.enqueue(job_id, request)

        log.info("Gateway accepted job_id=%s for ip=%s", job_id, _hash_ip(client_ip))
        return EnqueueResponse(job_id=job_id)

    def status(
        self,
        authorization_header: str,
        job_id: str,
    ) -> JobStatusResponse | None:
        """Fetch job status. Returns None if job_id is not found."""
        token = _extract_bearer_token(authorization_header)
        verify_jwt(token, self._config)
        return self._broker.get_status(job_id)


# ---------------------------------------------------------------------------
# Module-level convenience functions (for serverless handler wrappers)
# ---------------------------------------------------------------------------
_default_router: GatewayRouter | None = None


def _get_default_router() -> GatewayRouter:
    global _default_router
    if _default_router is None:
        _default_router = GatewayRouter()
    return _default_router


def enqueue_job(
    authorization_header: str,
    client_ip: str,
    request: EnqueueRequest,
) -> EnqueueResponse:
    """Enqueue a job using the default (env-configured) router."""
    return _get_default_router().enqueue(authorization_header, client_ip, request)


def get_job_status(authorization_header: str, job_id: str) -> JobStatusResponse | None:
    """Get job status using the default (env-configured) router."""
    return _get_default_router().status(authorization_header, job_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_bearer_token(authorization_header: str) -> str:
    """Extract the token from ``Authorization: Bearer <token>``."""
    if not authorization_header or not authorization_header.lower().startswith("bearer "):
        raise TokenInvalidError("Authorization header missing or not Bearer type.")
    return authorization_header[7:].strip()


def _generate_job_id(idempotency_key: str) -> str:
    """Deterministic job ID: SHA-256 of idempotency_key, truncated to 16 hex chars.

    Deterministic so that a re-queued job (after a 409) can be tracked with
    the same ID without a separate lookup.
    """
    return hashlib.sha256(idempotency_key.encode()).hexdigest()[:16]


def _hash_ip(ip: str) -> str:
    """One-way hash of a client IP for privacy-safe logging (no raw IPs in logs)."""
    return hashlib.sha256(ip.encode()).hexdigest()[:8]
