"""api.py — FastAPI serverless entry-point for the Edge Gateway.

Deployed as a Vercel Python serverless function (see vercel.json). The ASGI
app object is ``app``; Vercel's Python runtime detects it automatically.

Request lifecycle (target: < 50 ms p99, no serverless timeout possible)
-----------------------------------------------------------------------
    POST /v1/jobs
      1. Strict JWT validation (HS256, exp/iat/iss required, ≤ 15-min validity)
      2. Distributed rate limiting (Redis INCR sliding window, per-IP + per-sub)
      3. Idempotency claim (SET NX EX, 24 h)
      4. XADD to Redis Stream  →  202 Accepted { job_id }

    GET /v1/jobs/{job_id}
      1. Strict JWT validation
      2. HGETALL job:{job_id}  →  200 { status, ... } | 404

No AI calls, no database writes, no blocking work of any kind happens here.
The heavy lifting is done by the Processing Fleet consuming the stream.

Extreme error handling — the failure matrix
-------------------------------------------
| Failure                        | Response                | Why                          |
|--------------------------------|-------------------------|------------------------------|
| Missing/invalid/expired JWT    | 401 + WWW-Authenticate  | Auth is binary               |
| Rate limit exceeded            | 429 + Retry-After       | Client must back off         |
| Duplicate idempotency key      | 409 + existing job_id   | Caller already succeeded     |
| Redis DOWN during enqueue      | 503 + Retry-After       | Transient; client retries    |
| Redis DOWN during rate-check   | FAIL CLOSED → 503       | Never bypass a broken limiter|
| Malformed body                 | 422 (FastAPI/pydantic)  | Client bug                   |
| Anything unexpected            | 500, generic message    | Never leak internals         |

"Fail closed" on the rate limiter is a deliberate security decision: if the
limiter cannot be consulted, we refuse traffic rather than allow unmetered
abuse against paid AI provider keys.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import redis as redis_lib
from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..logging_setup import get_logger
from .gateway import (
    EnqueueRequest,
    EnqueueResponse,
    GatewayConfig,
    IdempotencyConflictError,
    JobStatusResponse,
    RateLimitExceededError,
    RedisJobBroker,
    TokenInvalidError,
    _generate_job_id,
    verify_jwt,
)

log = get_logger("gateway.api")

app = FastAPI(
    title="Autoposter Edge Gateway",
    version="1.0.0",
    docs_url=None,       # no public Swagger in production
    redoc_url=None,
    openapi_url=None,
)

_bearer = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Lazy singletons — built on first request, rebuilt after poisoning.
# Serverless containers are reused between invocations; caching the broker
# amortises the TLS handshake, but a failed client must never be cached.
# ---------------------------------------------------------------------------
_config: GatewayConfig | None = None
_broker: RedisJobBroker | None = None


def _get_config() -> GatewayConfig:
    global _config
    if _config is None:
        _config = GatewayConfig()  # raises KeyError only on missing env — caught below
    return _config


def _get_broker() -> RedisJobBroker:
    global _broker
    if _broker is None:
        _broker = RedisJobBroker(_get_config())
    return _broker


def _poison_broker() -> None:
    """Drop the cached broker after a connection-class failure so the next
    invocation builds a fresh client instead of reusing a dead socket."""
    global _broker
    _broker = None


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
async def _authenticated_claims(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, Any]:
    """Strict JWT validation dependency. Raises 401 on ANY failure mode."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized("Missing bearer token.")
    try:
        return verify_jwt(credentials.credentials, _get_config())
    except TokenInvalidError as exc:
        # Single generic message regardless of failure mode — no oracle.
        raise _unauthorized(str(exc))


def _client_ip(request: Request) -> str:
    """Resolve the caller IP behind Vercel's proxy (first x-forwarded-for hop)."""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _unauthorized(detail: str):
    from fastapi import HTTPException

    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post(
    "/v1/jobs",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=EnqueueResponse,
)
async def enqueue_job(
    body: EnqueueRequest,
    request: Request,
    claims: dict[str, Any] = Depends(_authenticated_claims),
) -> Response:
    """Accept a job, push to the broker, return 202 within milliseconds."""
    ip = _client_ip(request)
    sub = str(claims.get("sub", "anonymous"))
    broker = _get_broker()

    # ---- 2. Rate limiting (FAIL CLOSED on Redis failure) --------------------
    try:
        broker.check_rate_limit(ip)          # per-IP: network-level abuse
        broker.check_rate_limit(f"sub:{sub}")  # per-subject: token-level abuse
    except RateLimitExceededError as exc:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"error": "rate_limited", "detail": str(exc)},
            headers={"Retry-After": str(exc.window_seconds)},
        )
    except (redis_lib.ConnectionError, redis_lib.TimeoutError, OSError) as exc:
        _poison_broker()
        log.error("Rate limiter unreachable — failing closed: %s", type(exc).__name__)
        return _service_unavailable("Rate limiter unavailable; request refused.")

    # ---- 3. Idempotency ------------------------------------------------------
    try:
        broker.ensure_idempotency(body.idempotency_key)
    except IdempotencyConflictError:
        # Not an error for the caller: their earlier request already won.
        # Return the deterministic job_id so they can poll its status.
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error": "duplicate_request",
                "job_id": _generate_job_id(body.idempotency_key),
                "detail": "This idempotency key was already accepted. Poll the job_id for status.",
            },
        )
    except (redis_lib.ConnectionError, redis_lib.TimeoutError, OSError) as exc:
        _poison_broker()
        log.error("Redis unreachable during idempotency claim: %s", type(exc).__name__)
        return _service_unavailable("Job broker unavailable; retry with the same idempotency key.")

    # ---- 4. Enqueue ------------------------------------------------------------
    job_id = _generate_job_id(body.idempotency_key)
    try:
        broker.enqueue(job_id, body)
    except (redis_lib.ConnectionError, redis_lib.TimeoutError, OSError) as exc:
        _poison_broker()
        # CRITICAL EDGE CASE: idempotency key was claimed but the job was NOT
        # enqueued. Best-effort rollback of the claim so the caller's retry
        # (same key) is not spuriously rejected with 409 for 24 hours.
        try:
            broker._client.delete(f"idempotency:{body.idempotency_key}")  # noqa: SLF001
        except Exception:
            log.critical(
                "ORPHANED idempotency key %s — enqueue failed AND rollback failed. "
                "Manual reconciliation required.", body.idempotency_key,
            )
        log.error("Redis unreachable during enqueue: %s", type(exc).__name__)
        return _service_unavailable("Job broker unavailable; retry with the same idempotency key.")

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=EnqueueResponse(job_id=job_id).model_dump(),
        headers={"Location": f"/v1/jobs/{job_id}"},
    )


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(
    job_id: str,
    claims: dict[str, Any] = Depends(_authenticated_claims),
) -> Response:
    """Poll job status from the Redis status hash."""
    try:
        result = _get_broker().get_status(job_id)
    except (redis_lib.ConnectionError, redis_lib.TimeoutError, OSError) as exc:
        _poison_broker()
        log.error("Redis unreachable during status read: %s", type(exc).__name__)
        return _service_unavailable("Status store unavailable; retry shortly.")

    if result is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "not_found", "detail": f"No job with id {job_id!r}."},
        )
    return JSONResponse(status_code=status.HTTP_200_OK, content=result.model_dump())


@app.get("/v1/health")
async def health() -> dict[str, str]:
    """Liveness probe — deliberately does NOT touch Redis (that's readiness)."""
    return {"status": "ok", "ts": dt.datetime.now(dt.UTC).isoformat()}


# ---------------------------------------------------------------------------
# Global exception hardening
# ---------------------------------------------------------------------------
@app.exception_handler(KeyError)
async def _missing_env_handler(request: Request, exc: KeyError) -> JSONResponse:
    """A KeyError at this level means a required env var is missing (GatewayConfig).
    Return 503 with a generic message; log the specific var name server-side only."""
    log.critical("Gateway misconfigured — missing env var: %s", exc)
    return _service_unavailable("Gateway is not configured correctly.")


@app.exception_handler(Exception)
async def _catch_all_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last line of defence: never leak stack traces or internals to callers."""
    log.exception("Unhandled gateway exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_error", "detail": "An unexpected error occurred."},
    )


def _service_unavailable(detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"error": "service_unavailable", "detail": detail},
        headers={"Retry-After": "5"},
    )
