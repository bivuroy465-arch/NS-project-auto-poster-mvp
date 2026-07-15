"""Event Broker Consumer — Redis Streams XREADGROUP with DLQ and circuit breaker.

Design principles
-----------------
  * Consumer Groups (XREADGROUP/XACK/XAUTOCLAIM): each message is delivered to
    exactly one consumer in the group.  If the consumer process is OOM-killed
    before XACK, the message remains in the Pending Entry List (PEL) with its
    original delivery count incremented, and the next XAUTOCLAIM cycle on any
    live worker will reclaim it.

  * XACK is the exclusive responsibility of the *caller* (Worker._process_message).
    This module never ACKs — it is the single, non-negotiable rule that makes
    OOM-kill survival correct.  Even if this code is refactored, the docstring
    and the type signature enforce it: ``ConsumedMessage.ack()`` exists as an
    explicit method that the Worker must call.

  * Dead Letter Queue (DLQ): after ``ConsumerConfig.max_delivery_attempts``
    failures, ``route_to_dlq()`` atomically:
        1. XADD the message payload to the DLQ stream.
        2. XACK the original entry (remove from PEL so it is not re-delivered).
    The two operations are pipelined but NOT wrapped in MULTI/EXEC because XADD
    and XACK are idempotent: if the process crashes between the two calls, the
    next XAUTOCLAIM cycle re-delivers; the DLQ entry is a duplicate, which is
    acceptable for dead-letter monitoring.

  * Circuit Breaker (CLOSED → OPEN → HALF-OPEN):
      CLOSED   — normal operation.
      OPEN     — after ``circuit_fail_threshold`` consecutive Redis errors,
                 stops attempting connections for ``circuit_open_seconds``.
      HALF-OPEN — after the open timeout, allows one probe attempt.
                  Success → CLOSED.  Failure → OPEN again with reset timer.
    This prevents a flapping Redis connection from hammering reconnect logic
    and exhausting file descriptors.

  * Idle PEL reclaim (XAUTOCLAIM): on every ``reclaim_every_n_polls`` poll
    cycle, the consumer scans for messages that have been in the PEL longer
    than ``claim_idle_ms`` (default 5 min).  This handles the case where a
    worker was OOM-killed mid-execution and never called XACK.

  * Consumer group creation (MKSTREAM): idempotent; existing groups are
    detected via the ``BUSYGROUP`` error code and silently accepted.

Stream key schema (mirrors gateway.py constants)
-------------------------------------------------
    autoposter:jobs         — primary stream; workers XREADGROUP from here.
    autoposter:dlq          — dead letter queue; operator inspects manually.
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

import redis

from ..logging_setup import get_logger

log = get_logger("broker.consumer")

# ---------------------------------------------------------------------------
# Constants (kept in sync with gateway.py)
# ---------------------------------------------------------------------------
_STREAM_KEY: str = "autoposter:jobs"
_DLQ_STREAM: str = "autoposter:dlq"
_CONSUMER_GROUP: str = "fleet-workers"

# ---------------------------------------------------------------------------
# Circuit breaker states
# ---------------------------------------------------------------------------
_CB_CLOSED = "CLOSED"
_CB_OPEN = "OPEN"
_CB_HALF_OPEN = "HALF_OPEN"


class CircuitOpenError(RuntimeError):
    """Raised when the circuit breaker is open and a Redis call is blocked."""


# ---------------------------------------------------------------------------
# Consumed message handle
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ConsumedMessage:
    """A single stream entry returned to the Worker.

    The Worker is the exclusive owner of ``ack()`` and ``route_to_dlq()``.
    This class intentionally carries no Redis client reference — it is a pure
    data object.  The consumer passes itself as the *broker* so those calls
    are dispatched back through the circuit-breaker-protected connection.
    """

    stream_id: str          # Redis stream entry id, e.g. "1704067200000-0"
    job_id: str
    idempotency_key: str
    platform_overrides: list[str] | None
    dry_run: bool
    extra: dict[str, Any]
    delivery_count: int     # XPENDING delivery count (1 on first delivery)

    # Back-reference to the consumer that delivered this message.  Set by
    # StreamConsumer.poll(); not part of the serialised payload.
    _consumer: StreamConsumer = field(repr=False, compare=False)

    def ack(self) -> None:
        """XACK this entry, removing it from the PEL.

        MUST be called only after the job has been fully committed — never
        before.  If the worker process is killed before this call, the
        message stays in the PEL and will be reclaimed by XAUTOCLAIM.
        """
        self._consumer.ack(self.stream_id)

    def route_to_dlq(self, error: str) -> None:
        """Move this entry to the DLQ and XACK the original."""
        self._consumer.route_to_dlq(self, error)

    @classmethod
    def _parse(
        cls,
        stream_id: str,
        fields: dict[str, str],
        delivery_count: int,
        consumer: "StreamConsumer",
    ) -> "ConsumedMessage":
        payload = json.loads(fields["payload"])
        return cls(
            stream_id=stream_id,
            job_id=payload["job_id"],
            idempotency_key=payload["idempotency_key"],
            platform_overrides=payload.get("platform_overrides"),
            dry_run=bool(payload.get("dry_run", False)),
            extra=payload.get("extra", {}),
            delivery_count=delivery_count,
            _consumer=consumer,
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ConsumerConfig:
    redis_url: str = field(
        default_factory=lambda: os.environ.get("UPSTASH_REDIS_REST_URL", "redis://localhost:6379")
    )
    stream_key: str = _STREAM_KEY
    dlq_stream: str = _DLQ_STREAM
    consumer_group: str = _CONSUMER_GROUP
    consumer_name: str = field(
        default_factory=lambda: f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
    )
    # Polling
    poll_block_ms: int = 5_000          # XREADGROUP BLOCK duration
    reclaim_every_n_polls: int = 10     # run XAUTOCLAIM every N poll cycles
    claim_idle_ms: int = 300_000        # 5 min: reclaim msgs from dead workers
    reclaim_batch: int = 10             # XAUTOCLAIM count per cycle
    # Delivery / DLQ
    max_delivery_attempts: int = 3
    # Circuit breaker
    circuit_fail_threshold: int = 5     # consecutive errors before OPEN
    circuit_open_seconds: float = 30.0  # how long to stay OPEN


# ---------------------------------------------------------------------------
# StreamConsumer
# ---------------------------------------------------------------------------
class StreamConsumer:
    """Redis Streams XREADGROUP consumer with DLQ and circuit breaker.

    Designed for use by Worker.  The canonical usage pattern::

        consumer = StreamConsumer(config)
        consumer.ensure_group()          # idempotent, call at startup

        for msg in consumer.poll():      # blocks up to poll_block_ms
            try:
                process(msg)
                msg.ack()               # ONLY after successful commit
            except Unrecoverable:
                msg.route_to_dlq(str(e))

    The poll() method also periodically runs XAUTOCLAIM to reclaim idle
    messages from dead workers — callers do not need to do anything special.
    """

    def __init__(self, config: ConsumerConfig | None = None) -> None:
        self._cfg = config or ConsumerConfig()
        self._client: redis.Redis | None = None
        self._poll_count: int = 0

        # Circuit breaker state
        self._cb_state: str = _CB_CLOSED
        self._cb_failures: int = 0
        self._cb_opened_at: float = 0.0

        self._connect()

    # -- Connection / circuit breaker ----------------------------------------

    def _connect(self) -> None:
        """Create (or recreate) the Redis client.  Never raises."""
        try:
            self._client = redis.from_url(
                self._cfg.redis_url,
                decode_responses=True,
                socket_timeout=10,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            # Ping to fail-fast on a bad URL / network partition at startup.
            self._client.ping()
            self._cb_failures = 0
            self._cb_state = _CB_CLOSED
            log.info("Broker connected to Redis (consumer=%s).", self._cfg.consumer_name)
        except redis.RedisError as exc:
            self._client = None
            log.error("Broker Redis connect failed: %s", exc)

    def _guard(self) -> redis.Redis:
        """Return the Redis client, enforcing circuit breaker state.

        Transitions:
          CLOSED   → any call goes through.
          OPEN     → if still within open window, raise CircuitOpenError.
                     If window has elapsed, transition to HALF_OPEN.
          HALF_OPEN → allow one probe attempt.  On success → CLOSED.
                      On failure → re-OPEN with reset timer.
        """
        now = time.monotonic()

        if self._cb_state == _CB_OPEN:
            elapsed = now - self._cb_opened_at
            if elapsed < self._cfg.circuit_open_seconds:
                remaining = self._cfg.circuit_open_seconds - elapsed
                raise CircuitOpenError(
                    f"Circuit breaker OPEN for {remaining:.1f}s more "
                    f"after {self._cb_failures} consecutive Redis failures."
                )
            # Enough time has passed — allow one probe.
            log.info("Circuit breaker: OPEN → HALF_OPEN (probe attempt).")
            self._cb_state = _CB_HALF_OPEN

        if self._client is None:
            self._connect()
            if self._client is None:
                self._record_failure()
                raise CircuitOpenError("Redis client is not available.")

        return self._client

    def _record_success(self) -> None:
        if self._cb_state in (_CB_HALF_OPEN, _CB_OPEN):
            log.info("Circuit breaker: %s → CLOSED (probe succeeded).", self._cb_state)
        self._cb_state = _CB_CLOSED
        self._cb_failures = 0

    def _record_failure(self) -> None:
        self._cb_failures += 1
        if self._cb_failures >= self._cfg.circuit_fail_threshold or self._cb_state == _CB_HALF_OPEN:
            self._cb_state = _CB_OPEN
            self._cb_opened_at = time.monotonic()
            self._client = None  # force reconnect on next HALF_OPEN probe
            log.error(
                "Circuit breaker: → OPEN after %d failures. "
                "Will retry in %.0fs.",
                self._cb_failures,
                self._cfg.circuit_open_seconds,
            )
        else:
            self._cb_state = _CB_CLOSED  # not yet at threshold; remain closed

    # -- Group management ----------------------------------------------------

    def ensure_group(self) -> None:
        """Create the consumer group if it does not already exist (idempotent).

        Uses MKSTREAM so the stream itself is created if absent — this means
        the gateway and the worker can start in either order.
        """
        client = self._guard()
        try:
            client.xgroup_create(
                self._cfg.stream_key,
                self._cfg.consumer_group,
                id="0",         # deliver all existing entries on first start
                mkstream=True,  # create the stream if it doesn't exist
            )
            log.info(
                "Consumer group %r created on stream %r.",
                self._cfg.consumer_group,
                self._cfg.stream_key,
            )
        except redis.exceptions.ResponseError as exc:
            if "BUSYGROUP" in str(exc):
                log.debug("Consumer group %r already exists.", self._cfg.consumer_group)
            else:
                raise

    # -- Polling -------------------------------------------------------------

    def poll(self) -> Iterator[ConsumedMessage]:
        """XREADGROUP blocking poll.  Yields at most one message per call.

        Callers iterate in a while-loop::

            while running:
                for msg in consumer.poll():
                    ...

        Returns an empty iterator on timeout (no messages available).
        Raises ``CircuitOpenError`` if the circuit is open.
        Raises ``redis.RedisError`` for unrecoverable errors (caller should
        catch and back off).
        """
        self._poll_count += 1
        client = self._guard()

        try:
            # ">" = only messages that have never been delivered to any consumer.
            entries = client.xreadgroup(
                groupname=self._cfg.consumer_group,
                consumername=self._cfg.consumer_name,
                streams={self._cfg.stream_key: ">"},
                count=1,
                block=self._cfg.poll_block_ms,
            )
            self._record_success()
        except redis.RedisError as exc:
            self._record_failure()
            raise

        if entries:
            for _stream, messages in entries:
                for stream_id, fields in messages:
                    try:
                        # delivery_count for fresh messages is 1; XPENDING
                        # would give the real count but costs an extra round-trip.
                        # The worker increments its own counter in the job hash,
                        # which is the authoritative retry count.
                        yield ConsumedMessage._parse(stream_id, fields, 1, self)
                    except (KeyError, json.JSONDecodeError, TypeError) as exc:
                        log.error(
                            "Malformed stream entry %s — routing to DLQ: %s",
                            stream_id, exc,
                        )
                        self._dlq_raw(stream_id, fields, str(exc))

        # Periodic idle PEL reclaim (handles OOM-killed workers).
        if self._poll_count % self._cfg.reclaim_every_n_polls == 0:
            yield from self._reclaim_idle()

    # -- PEL reclaim (dead-worker recovery) ----------------------------------

    def _reclaim_idle(self) -> Iterator[ConsumedMessage]:
        """XAUTOCLAIM: atomically steal messages idle > claim_idle_ms."""
        try:
            client = self._guard()
            result = client.xautoclaim(
                self._cfg.stream_key,
                self._cfg.consumer_group,
                self._cfg.consumer_name,
                min_idle_time=self._cfg.claim_idle_ms,
                start_id="0-0",
                count=self._cfg.reclaim_batch,
            )
            # result: (next_start_id, [(stream_id, fields), ...], [deleted_ids])
            claimed = result[1] if result else []
            if claimed:
                log.info(
                    "XAUTOCLAIM: reclaimed %d idle messages from dead workers.",
                    len(claimed),
                )
            for stream_id, fields in claimed:
                if not fields:
                    # Deleted entry (no longer in stream) — skip.
                    continue
                try:
                    # Fetch true delivery count from XPENDING for reclaimed msgs.
                    delivery_count = self._get_delivery_count(stream_id)
                    yield ConsumedMessage._parse(stream_id, fields, delivery_count, self)
                except (KeyError, json.JSONDecodeError, TypeError) as exc:
                    log.error("Malformed reclaimed entry %s — DLQ: %s", stream_id, exc)
                    self._dlq_raw(stream_id, fields, str(exc))
        except (redis.RedisError, CircuitOpenError) as exc:
            log.warning("XAUTOCLAIM skipped (non-critical): %s", exc)

    def _get_delivery_count(self, stream_id: str) -> int:
        """XPENDING lookup for a single entry's delivery count."""
        try:
            client = self._guard()
            pending = client.xpending_range(
                self._cfg.stream_key,
                self._cfg.consumer_group,
                min=stream_id,
                max=stream_id,
                count=1,
            )
            if pending:
                return int(pending[0].get("times_delivered", 1))
        except (redis.RedisError, CircuitOpenError):
            pass
        return 1  # safe fallback

    # -- Ack / DLQ -----------------------------------------------------------

    def ack(self, stream_id: str) -> None:
        """XACK a message.  Only call after the job has been fully committed."""
        try:
            client = self._guard()
            client.xack(self._cfg.stream_key, self._cfg.consumer_group, stream_id)
            self._record_success()
            log.debug("XACK stream_id=%s", stream_id)
        except (redis.RedisError, CircuitOpenError) as exc:
            # Non-fatal: the message stays in the PEL and will be reclaimed.
            # Log at error level — an operator should investigate.
            log.error(
                "XACK FAILED for stream_id=%s: %s — message will be reclaimed and reprocessed.",
                stream_id, exc,
            )

    def route_to_dlq(self, msg: ConsumedMessage, error: str) -> None:
        """Append to DLQ and XACK the original entry."""
        self._dlq_raw(
            msg.stream_id,
            {
                "job_id": msg.job_id,
                "idempotency_key": msg.idempotency_key,
                "platform_overrides": json.dumps(msg.platform_overrides),
                "dry_run": str(msg.dry_run),
                "extra": json.dumps(msg.extra),
                "delivery_count": str(msg.delivery_count),
            },
            error,
        )

    def _dlq_raw(
        self,
        original_stream_id: str,
        original_fields: dict[str, str],
        error: str,
    ) -> None:
        """Low-level DLQ write + XACK.  Pipeline (not MULTI/EXEC) for performance.

        MULTI/EXEC is intentionally NOT used.  Both XADD and XACK are
        idempotent operations: if the process crashes between them, the
        worst case is a duplicate DLQ entry, which is safe for monitoring.
        Using MULTI/EXEC would block other clients and doesn't add correctness.
        """
        try:
            client = self._guard()
            pipe = client.pipeline(transaction=False)
            pipe.xadd(self._cfg.dlq_stream, {
                "original_stream_id": original_stream_id,
                "error": error[:2000],
                "payload": json.dumps(original_fields),
            })
            pipe.xack(
                self._cfg.stream_key,
                self._cfg.consumer_group,
                original_stream_id,
            )
            pipe.execute()
            self._record_success()
            log.error(
                "DLQ: stream_id=%s error=%s",
                original_stream_id, error[:200],
            )
        except (redis.RedisError, CircuitOpenError) as exc:
            # Last-resort: at least log the failure.  Message will remain in
            # PEL and be reclaimed — another worker will DLQ it.
            log.critical(
                "FAILED to write DLQ entry for stream_id=%s: %s",
                original_stream_id, exc,
            )

    # -- Convenience ---------------------------------------------------------

    @property
    def circuit_state(self) -> str:
        return self._cb_state
