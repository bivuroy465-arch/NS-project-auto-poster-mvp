"""Processing Fleet Worker — the critical-path orchestrator.

Responsibility
--------------
The Worker is the composition root that wires together:

  StreamConsumer  — claims one Redis Stream message per poll cycle.
  DistributedLock — prevents concurrent execution of the same job on two workers.
  CredentialRepository.inject() — JIT decryption; plaintext lives in one call frame.
  main.run(cfg)   — the existing pipeline; receives a per-job config dict.
  OTel tracer     — tracks processing time; strictly scrubs all API key material.

OOM-kill contract
-----------------
Redis XACK is called in *exactly one place*: after the pipeline has returned
successfully and the distributed lock is about to be released.  There is no
``ack()`` call in any ``except`` or ``finally`` block.  If the process is
OOM-killed at any point during execution:

  1. The stream entry stays in the PEL (Pending Entry List) — never lost.
  2. After claim_idle_ms (5 min), XAUTOCLAIM on another worker reclaims it.
  3. The reclaimed entry has delivery_count ≥ 2.  When it reaches
     max_delivery_attempts, it is moved to the DLQ by route_to_dlq().
  4. The distributed lock TTL (330 s) expires automatically, so the new
     worker can acquire it.

This is the single guarantee that makes the system resilient to catastrophic
worker failure: XACK is the *last* action on the success path, never the
first, and it is *never* called on the failure path.

Thread / process model
-----------------------
The Worker is designed to run as a single-threaded blocking loop in its own
OS process.  Scale out by launching more identical processes (Kubernetes
replicas, Cloud Run min-instances > 1, etc.).  Redis Consumer Groups handle
distribution automatically.

Idempotency layers (defence-in-depth)
--------------------------------------
  Layer 1 — Gateway:  ``SET NX`` on idempotency_key before XADD.
            Prevents duplicate enqueues at the HTTP layer.
  Layer 2 — Worker:   ``SET NX`` on ``idempotency:processed:{key}`` (24h TTL)
            before acquiring the lock.  If two workers race past Layer 1
            (e.g. a 409 that wasn't properly handled), the first one to SET
            proceeds; the second skips and XACK's immediately.
  Layer 3 — Lock:     Distributed lock prevents *concurrent* execution.
            Combined with Layer 2, this means a job executes at most once
            even if delivered multiple times.
"""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Any

import redis

from ..broker.consumer import CircuitOpenError, ConsumedMessage, ConsumerConfig, StreamConsumer
from ..config_loader import load_config
from ..fleet.lock import DistributedLock, LockConfig, LockNotAcquiredError
from ..gateway.gateway import JobStatus, _IDEMPOTENCY_PREFIX, _JOB_HASH_PREFIX
from ..logging_setup import get_logger

log = get_logger("fleet.worker")

_MAX_DELIVERY_ATTEMPTS = 3
_IDEMPOTENCY_PROCESSED_TTL = 86_400  # 24 hours


# ---------------------------------------------------------------------------
# Worker configuration
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class WorkerConfig:
    redis_url: str = field(
        default_factory=lambda: os.environ.get("UPSTASH_REDIS_REST_URL", "redis://localhost:6379")
    )
    worker_id: str = field(
        default_factory=lambda: _default_worker_id()
    )
    max_delivery_attempts: int = _MAX_DELIVERY_ATTEMPTS


def _default_worker_id() -> str:
    import socket
    import uuid
    return f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# DLQError
# ---------------------------------------------------------------------------
class DLQError(Exception):
    """Raised internally when a job must be sent to the DLQ."""


# ---------------------------------------------------------------------------
# JobMessage (re-exported for backwards compatibility with __init__.py)
# ---------------------------------------------------------------------------
# The canonical type is now ConsumedMessage from src.broker.consumer.
# JobMessage is kept as a thin alias so existing code that imports
# ``from src.fleet import JobMessage`` continues to work.
JobMessage = ConsumedMessage


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
class Worker:
    """Redis Stream consumer worker.

    Lifecycle::

        worker = Worker(config)
        worker.start()   # registers signal handlers, creates consumer group
        worker.run()     # blocking poll loop
        worker.stop()    # graceful shutdown (also called on SIGTERM/SIGINT)
    """

    def __init__(
        self,
        config: WorkerConfig | None = None,
        enclave_manager: Any | None = None,
    ) -> None:
        self._cfg = config or WorkerConfig()

        # Shared Redis client for status/idempotency writes.
        self._redis = redis.from_url(
            self._cfg.redis_url,
            decode_responses=True,
            socket_timeout=10,
            socket_connect_timeout=5,
            retry_on_timeout=True,
        )

        # Event broker consumer (its own connection + circuit breaker).
        self._consumer = StreamConsumer(
            ConsumerConfig(
                redis_url=self._cfg.redis_url,
                consumer_name=self._cfg.worker_id,
            )
        )

        # Distributed lock (Redlock-pattern; single-node by default).
        self._lock = DistributedLock(
            worker_id=self._cfg.worker_id,
            config=LockConfig(
                redis_urls=[self._cfg.redis_url],
            ),
        )

        self._enclave = enclave_manager
        self._running = False

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Register signal handlers and ensure the consumer group exists."""
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        self._consumer.ensure_group()
        self._running = True
        log.info("Worker %s started.", self._cfg.worker_id)

    def stop(self) -> None:
        """Request graceful shutdown; the current job will finish before exit."""
        log.info("Worker %s: stop requested — finishing current job.", self._cfg.worker_id)
        self._running = False

    def run(self) -> None:
        """Blocking poll loop.  Returns only when ``stop()`` is called."""
        while self._running:
            try:
                for msg in self._consumer.poll():
                    self._process_message(msg)
            except CircuitOpenError as exc:
                log.warning("Broker circuit open: %s — sleeping 10s.", exc)
                time.sleep(10)
            except redis.RedisError as exc:
                log.error("Redis error in poll loop: %s — sleeping 5s.", exc)
                time.sleep(5)
            except Exception:
                log.exception("Unexpected error in poll loop — sleeping 5s.")
                time.sleep(5)
        log.info("Worker %s shut down cleanly.", self._cfg.worker_id)

    # -- Critical path -------------------------------------------------------

    def _process_message(self, msg: ConsumedMessage) -> None:
        """Full job lifecycle.

        ┌─────────────────────────────────────────────────────┐
        │  1. Idempotency guard (SET NX — layer 2)            │
        │  2. Acquire distributed lock                        │
        │  3. Mark status = PROCESSING                        │
        │  4. _execute_pipeline()                             │
        │  5. XACK  ← only here on success                   │
        │  6. Mark status = COMPLETE                          │
        │  7. Release lock (context manager __exit__)         │
        └─────────────────────────────────────────────────────┘
        Failure path:
          * delivery_count < max_delivery_attempts:
              leave in PEL — XAUTOCLAIM will reclaim after idle timeout.
          * delivery_count >= max_delivery_attempts:
              msg.route_to_dlq() → XADD dlq + XACK original.
        """
        log.info("Processing job_id=%s stream_id=%s", msg.job_id, msg.stream_id)

        # --- Layer 2 idempotency guard -----------------------------------
        if not self._check_and_mark_idempotency(msg.idempotency_key):
            log.info(
                "Job %s already processed (idempotency key seen) — skipping.",
                msg.job_id,
            )
            msg.ack()
            self._update_status(msg.job_id, JobStatus.COMPLETE, result={"skipped": "duplicate"})
            return

        # --- Distributed lock acquisition --------------------------------
        handle = self._lock.try_acquire(msg.job_id)
        if handle is None:
            # Another worker is processing this job right now.
            # XACK to remove from our PEL; the other worker will complete it.
            log.warning(
                "Lock for job_id=%s held by another worker — XACK and skip.",
                msg.job_id,
            )
            msg.ack()
            return

        # Lock is held.  Everything from here until lock release is the
        # exclusive critical section.
        with handle:
            try:
                # --- Mark as processing ----------------------------------
                self._update_status(msg.job_id, JobStatus.PROCESSING)

                # --- Execute pipeline ------------------------------------
                self._execute_pipeline(msg)

                # --- Success: XACK is the last action -------------------
                # This is the ONLY call site for msg.ack() on the success path.
                # If the process is OOM-killed before reaching this line, the
                # message remains in the PEL for reclaim.
                msg.ack()
                self._update_status(msg.job_id, JobStatus.COMPLETE, result={"published": True})
                log.info("Job %s completed successfully.", msg.job_id)

            except Exception as exc:
                self._handle_failure(msg, exc)
                # NOTE: msg.ack() is NOT called here.  The message stays in
                # the PEL so it can be reclaimed and retried (or DLQ'd on
                # the final attempt).

    # -- Pipeline execution --------------------------------------------------

    def _execute_pipeline(self, msg: ConsumedMessage) -> None:
        """JIT credential injection → pipeline execution → OTel span.

        Credentials are injected into ``os.environ`` for the duration of
        ``main.run()`` and cleared in a ``finally`` block regardless of
        outcome.  The enclave's own ``inject()`` method also zeroes the
        plaintext buffer, so the value exists in memory for the minimum
        possible window.

        API keys are NEVER passed as span attributes, logged, or included
        in exception messages.  The RedactingSpanProcessor provides a
        second line of defence for any attribute that does leak through.
        """
        # Build per-job config dict.
        base_cfg = load_config()
        if msg.platform_overrides:
            base_cfg["platforms"] = msg.platform_overrides
        if msg.dry_run:
            base_cfg["dry_run"] = True
        base_cfg.update(msg.extra)

        # JIT credential injection: populate os.environ for the pipeline's
        # duration, then clear.  If no enclave is configured, credentials
        # are already expected to be in the environment (CI / local mode).
        injected_keys: list[str] = []
        if self._enclave is not None:
            _INJECTABLE = [
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GEMINI_API_KEY",
                "GROQ_API_KEY",
                "STABILITY_API_KEY",
                "TWITTER_API_KEY",
                "TWITTER_API_SECRET",
                "TWITTER_ACCESS_TOKEN",
                "TWITTER_ACCESS_SECRET",
                "LINKEDIN_ACCESS_TOKEN",
                "FACEBOOK_PAGE_ACCESS_TOKEN",
            ]
            for key_name in _INJECTABLE:
                if self._enclave._store.has(key_name):
                    self._enclave.inject_env(key_name)
                    injected_keys.append(key_name)

        # Import lazily: keeps worker startup fast; avoids importing the
        # entire AI / social SDK stack before any job is claimed.
        from .. import main as pipeline_main
        from ..observability import record_exception, tracer

        try:
            with tracer("fleet").start_as_current_span("pipeline.execute") as span:
                # Only non-sensitive attributes (see telemetry.py docstring).
                span.set_attribute("gateway.job_id", msg.job_id)
                span.set_attribute("fleet.worker_id", self._cfg.worker_id)
                span.set_attribute("fleet.attempt_number", msg.delivery_count)
                span.set_attribute("pipeline.dry_run", bool(base_cfg.get("dry_run")))
                span.set_attribute(
                    "pipeline.platforms",
                    ",".join(base_cfg.get("platforms", [])),
                )
                try:
                    exit_code = pipeline_main.run(base_cfg)
                except Exception as exc:
                    record_exception(span, exc)
                    raise
                if exit_code != 0:
                    err = RuntimeError(
                        f"Pipeline exited with non-zero code {exit_code} "
                        f"for job_id={msg.job_id}"
                    )
                    record_exception(span, err)
                    raise err
        finally:
            # Always clear injected keys regardless of pipeline outcome.
            for key_name in injected_keys:
                os.environ.pop(key_name, None)

    # -- Failure handling ----------------------------------------------------

    def _handle_failure(self, msg: ConsumedMessage, exc: Exception) -> None:
        """On failure: increment attempt counter, then DLQ or leave in PEL."""
        attempt = self._increment_attempts(msg.job_id)
        log.error(
            "Job %s failed on delivery %d/%d: %s",
            msg.job_id,
            attempt,
            self._cfg.max_delivery_attempts,
            exc,
        )
        if attempt >= self._cfg.max_delivery_attempts:
            # Final attempt exhausted — route to DLQ.
            # route_to_dlq() calls XADD(dlq) + XACK(original) atomically.
            msg.route_to_dlq(str(exc)[:2000])
            self._update_status(msg.job_id, JobStatus.DLQ, error=str(exc)[:2000])
            log.error("Job %s routed to DLQ after %d attempts.", msg.job_id, attempt)
        else:
            # Leave message in PEL — XAUTOCLAIM will re-deliver after idle timeout.
            self._update_status(msg.job_id, JobStatus.FAILED, error=str(exc)[:2000])
            log.info(
                "Job %s left in PEL for retry (attempt %d/%d).",
                msg.job_id,
                attempt,
                self._cfg.max_delivery_attempts,
            )

    # -- Redis helpers -------------------------------------------------------

    def _check_and_mark_idempotency(self, idempotency_key: str) -> bool:
        """Atomically claim the processed key.

        Returns True if the job should proceed (key was new).
        Returns False if the job was already processed (key existed — duplicate).
        """
        redis_key = f"{_IDEMPOTENCY_PREFIX}processed:{idempotency_key}"
        result = self._redis.set(redis_key, "1", nx=True, ex=_IDEMPOTENCY_PROCESSED_TTL)
        return result is not None

    def _update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Update the job status hash (best-effort; never raises)."""
        try:
            import datetime as dt
            hash_key = f"{_JOB_HASH_PREFIX}{job_id}"
            mapping: dict[str, str] = {"status": str(status)}
            if status in (JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.DLQ):
                mapping["completed_at"] = dt.datetime.now(dt.UTC).isoformat()
            if error:
                mapping["error"] = error
            if result:
                mapping["result"] = json.dumps(result)
            self._redis.hset(hash_key, mapping=mapping)
        except Exception as exc:
            log.warning("Status update failed for job %s: %s", job_id, exc)

    def _increment_attempts(self, job_id: str) -> int:
        """Atomically increment and return the delivery attempt counter."""
        try:
            hash_key = f"{_JOB_HASH_PREFIX}{job_id}"
            return int(self._redis.hincrby(hash_key, "attempts", 1))
        except Exception:
            return self._cfg.max_delivery_attempts  # safe default → DLQ

    def _handle_shutdown(self, signum: int, frame: object) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Convenience entrypoint
# ---------------------------------------------------------------------------
def run_worker(config: WorkerConfig | None = None) -> None:
    """Start a Worker and block until shutdown signal."""
    worker = Worker(config)
    worker.start()
    worker.run()


if __name__ == "__main__":
    run_worker()
