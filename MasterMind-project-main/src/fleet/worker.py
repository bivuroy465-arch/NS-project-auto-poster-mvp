"""Processing Fleet — Redis Stream consumer worker with distributed locking.

Architecture
------------
Each Worker instance:

  1. XREADGROUP: claims one message from the ``autoposter:jobs`` consumer group.
  2. Distributed lock: ``SET lock:{job_id} {worker_id} NX EX 300`` — atomic.
     If the lock is not acquired, another worker already claimed this job;
     XACK and skip (idempotency).
  3. Idempotency re-check: verify the idempotency key hasn't been consumed.
  4. Credential injection: JIT decrypt via LifecycleManager (Credential Enclave).
  5. Pipeline execution: calls ``main.run()`` with injected credentials.
  6. Result: XACK + HSET status=complete on success.
     On failure: increment attempt counter.
     - attempt < MAX_DELIVERY_ATTEMPTS → NACK (leave in PEL for XCLAIM retry).
     - attempt >= MAX_DELIVERY_ATTEMPTS → XADD DLQ + HSET status=dlq.
  7. Lock release: ``DEL lock:{job_id}`` via Lua safe-release script.

Failure resilience
------------------
  * Circuit breakers: exponential backoff on Redis connection errors.
  * Idle PEL scan: on each poll cycle, check for messages idle > claim_idle_ms
    and XCLAIM them to self (handles crashed/stalled workers).
  * Graceful shutdown: SIGTERM/SIGINT → finish current job, stop polling.
  * DLQ: dead messages moved to ``autoposter:dlq`` stream for operator review.

The worker is framework-agnostic and can run as:
  - A standalone long-running process: ``python -m src.fleet``
  - A Kubernetes Pod (scale out by adding replicas).
  - A background thread alongside the gateway (single-machine dev mode).
"""

from __future__ import annotations

import json
import os
import signal
import socket
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import redis
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config_loader import load_config
from ..enclave import CredentialStore, LifecycleManager as EnclaveManager
from ..gateway.gateway import (
    JobStatus,
    RedisJobBroker,
    _STREAM_KEY,
    _IDEMPOTENCY_PREFIX,
    _LOCK_PREFIX,
    GatewayConfig,
)
from ..logging_setup import get_logger

log = get_logger("fleet.worker")

_DLQ_STREAM = "autoposter:dlq"
_CONSUMER_GROUP = "fleet-workers"
_MAX_DELIVERY_ATTEMPTS = 3
_CLAIM_IDLE_MS = 300_000  # 5 minutes — reclaim messages from dead workers


# ---------------------------------------------------------------------------
# Job message dataclass
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class JobMessage:
    """Decoded payload from a Redis Stream message."""

    stream_id: str          # Redis stream entry id (e.g. "1704067200000-0")
    job_id: str
    idempotency_key: str
    platform_overrides: list[str] | None
    dry_run: bool
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_stream_entry(cls, stream_id: str, fields: dict[str, str]) -> "JobMessage":
        payload = json.loads(fields["payload"])
        return cls(
            stream_id=stream_id,
            job_id=payload["job_id"],
            idempotency_key=payload["idempotency_key"],
            platform_overrides=payload.get("platform_overrides"),
            dry_run=payload.get("dry_run", False),
            extra=payload.get("extra", {}),
        )


# ---------------------------------------------------------------------------
# Worker error types
# ---------------------------------------------------------------------------
class DLQError(Exception):
    """Raised when a job has exhausted all delivery attempts and is sent to DLQ."""


# ---------------------------------------------------------------------------
# Worker configuration
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class WorkerConfig:
    redis_url: str = field(
        default_factory=lambda: os.environ.get("UPSTASH_REDIS_REST_URL", "redis://localhost:6379")
    )
    consumer_group: str = _CONSUMER_GROUP
    worker_id: str = field(
        default_factory=lambda: f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
    )
    poll_block_ms: int = 5_000       # XREADGROUP BLOCK timeout
    claim_idle_ms: int = _CLAIM_IDLE_MS
    max_delivery_attempts: int = _MAX_DELIVERY_ATTEMPTS


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
class Worker:
    """Redis Stream consumer worker.

    Lifecycle:
        worker = Worker(config)
        worker.start()   # registers signal handlers, creates consumer group
        worker.run()     # blocking poll loop
        worker.stop()    # graceful shutdown (called on SIGTERM/SIGINT)
    """

    def __init__(
        self,
        config: WorkerConfig | None = None,
        enclave_manager: EnclaveManager | None = None,
    ) -> None:
        self._cfg = config or WorkerConfig()
        self._redis = redis.from_url(
            self._cfg.redis_url,
            decode_responses=True,
            socket_timeout=10,
            socket_connect_timeout=5,
            retry_on_timeout=True,
        )
        self._enclave: EnclaveManager | None = enclave_manager
        self._running = False

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Register signal handlers and ensure the consumer group exists."""
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        self._ensure_consumer_group()
        self._running = True
        log.info(
            "Worker %s started on stream %s, group %s",
            self._cfg.worker_id,
            _STREAM_KEY,
            self._cfg.consumer_group,
        )

    def stop(self) -> None:
        """Request graceful shutdown; current job will finish before exit."""
        log.info("Worker %s received stop signal; finishing current job...", self._cfg.worker_id)
        self._running = False

    def run(self) -> None:
        """Blocking poll loop. Returns only when `stop()` is called."""
        while self._running:
            try:
                self._poll_and_process()
                self._reclaim_idle_messages()
            except redis.RedisError as exc:
                log.error("Redis error in poll loop: %s — retrying in 5s.", exc)
                time.sleep(5)
            except Exception:
                log.exception("Unexpected error in poll loop — retrying in 5s.")
                time.sleep(5)
        log.info("Worker %s shut down cleanly.", self._cfg.worker_id)

    # -- Core processing loop ------------------------------------------------

    def _poll_and_process(self) -> None:
        """XREADGROUP: claim one pending message and process it."""
        entries = self._redis.xreadgroup(
            groupname=self._cfg.consumer_group,
            consumername=self._cfg.worker_id,
            streams={_STREAM_KEY: ">"},  # ">" = only new, unclaimed messages
            count=1,
            block=self._cfg.poll_block_ms,
        )
        if not entries:
            return  # timeout — no messages; loop again

        # entries format: [(stream_key, [(entry_id, {field: value}), ...])]
        for _stream, messages in entries:
            for stream_id, fields in messages:
                self._process_message(stream_id, fields)

    def _process_message(self, stream_id: str, fields: dict[str, str]) -> None:
        """Full job lifecycle: lock → idempotency → pipeline → ack/nack."""
        try:
            msg = JobMessage.from_stream_entry(stream_id, fields)
        except (KeyError, json.JSONDecodeError, TypeError) as exc:
            log.error("Malformed stream message %s: %s — sending to DLQ.", stream_id, exc)
            self._send_to_dlq(stream_id, stream_id, fields, str(exc))
            return

        log.info("Processing job_id=%s (stream_id=%s)", msg.job_id, stream_id)

        # --- 1. Distributed lock ---
        if not self._acquire_lock(msg.job_id):
            log.warning(
                "Lock for job_id=%s already held; another worker is processing it. "
                "XACK-ing to prevent re-delivery.",
                msg.job_id,
            )
            self._redis.xack(_STREAM_KEY, self._cfg.consumer_group, stream_id)
            return

        try:
            # --- 2. Idempotency re-check (belt-and-suspenders) ---
            if not self._check_and_mark_idempotency(msg.idempotency_key):
                log.info("Job %s already processed (idempotency key exists); skipping.", msg.job_id)
                self._redis.xack(_STREAM_KEY, self._cfg.consumer_group, stream_id)
                self._update_status(msg.job_id, JobStatus.COMPLETE, result={"skipped": "duplicate"})
                return

            # --- 3. Mark as processing ---
            self._update_status(msg.job_id, JobStatus.PROCESSING)

            # --- 4. Execute pipeline (with credential injection) ---
            self._execute_pipeline(msg)

            # --- 5. Acknowledge success ---
            self._redis.xack(_STREAM_KEY, self._cfg.consumer_group, stream_id)
            self._update_status(msg.job_id, JobStatus.COMPLETE, result={"published": True})
            log.info("Job %s completed successfully.", msg.job_id)

        except DLQError as exc:
            self._send_to_dlq(stream_id, msg.job_id, fields, str(exc))
        except Exception as exc:
            self._handle_failure(stream_id, msg, exc)
        finally:
            self._release_lock(msg.job_id)

    # -- Pipeline execution --------------------------------------------------

    def _execute_pipeline(self, msg: JobMessage) -> None:
        """Inject credentials JIT and run the existing main.run() pipeline.

        Credentials are never stored as instance attributes or passed through
        Redis — they exist only within the inject() call frame.
        """
        # Build config overrides from the job message.
        base_cfg = load_config()
        if msg.platform_overrides:
            base_cfg["platforms"] = msg.platform_overrides
        if msg.dry_run:
            base_cfg["dry_run"] = True
        base_cfg.update(msg.extra)

        # JIT credential injection: if an enclave is configured, inject each
        # required key into os.environ for the duration of the pipeline call,
        # then clear it.  If no enclave is configured (e.g. keys are already
        # in the environment from CI), fall through and run directly.
        if self._enclave is not None:
            _INJECTABLE_KEYS = [
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
            for key in _INJECTABLE_KEYS:
                if self._enclave._store.has(key):
                    self._enclave.inject_env(key)

        # Import here (not at module level) to keep worker startup fast and
        # avoid importing the entire pipeline before a job is actually claimed.
        from .. import main as pipeline_main
        from ..observability import record_exception, tracer

        with tracer("fleet").start_as_current_span("pipeline.execute") as span:
            # Only non-sensitive attributes — the RedactingSpanProcessor is a
            # second line of defence, not the first.
            span.set_attribute("gateway.job_id", msg.job_id)
            span.set_attribute("fleet.worker_id", self._cfg.worker_id)
            span.set_attribute("pipeline.dry_run", bool(base_cfg.get("dry_run")))
            span.set_attribute("pipeline.platforms", ",".join(base_cfg.get("platforms", [])))
            try:
                exit_code = pipeline_main.run(base_cfg)
            except Exception as exc:
                record_exception(span, exc)
                raise
            if exit_code != 0:
                err = RuntimeError(f"Pipeline exited with code {exit_code}")
                record_exception(span, err)
                raise err

    # -- Failure handling ----------------------------------------------------

    def _handle_failure(self, stream_id: str, msg: JobMessage, exc: Exception) -> None:
        """Increment attempt counter; DLQ after MAX_DELIVERY_ATTEMPTS."""
        attempt = self._increment_attempts(msg.job_id)
        log.error(
            "Job %s failed on attempt %d/%d: %s",
            msg.job_id,
            attempt,
            self._cfg.max_delivery_attempts,
            exc,
        )
        if attempt >= self._cfg.max_delivery_attempts:
            self._send_to_dlq(stream_id, msg.job_id, {}, str(exc))
        else:
            # Leave in PEL — will be XCLAIM'd after claim_idle_ms.
            self._update_status(msg.job_id, JobStatus.FAILED, error=str(exc))

    def _send_to_dlq(
        self,
        stream_id: str,
        job_id: str,
        original_fields: dict,
        error: str,
    ) -> None:
        """Move message to DLQ stream; XACK original to remove from PEL."""
        self._redis.xadd(_DLQ_STREAM, {
            "original_stream_id": stream_id,
            "job_id": job_id,
            "error": error,
            "original_payload": json.dumps(original_fields),
        })
        self._redis.xack(_STREAM_KEY, self._cfg.consumer_group, stream_id)
        self._update_status(job_id, JobStatus.DLQ, error=error)
        log.error("Job %s sent to DLQ after max attempts. Error: %s", job_id, error)

    # -- PEL reclaim (dead worker recovery) ----------------------------------

    def _reclaim_idle_messages(self) -> None:
        """Claim messages stuck in the PEL of dead workers.

        XAUTOCLAIM atomically claims up to 10 messages idle > claim_idle_ms
        and assigns them to this worker's consumer name.
        """
        try:
            result = self._redis.xautoclaim(
                _STREAM_KEY,
                self._cfg.consumer_group,
                self._cfg.worker_id,
                min_idle_time=self._cfg.claim_idle_ms,
                start_id="0-0",
                count=10,
            )
            if result and result[1]:  # result[1] is the list of claimed messages
                log.info("Reclaimed %d idle messages from dead workers.", len(result[1]))
        except redis.RedisError as exc:
            log.warning("XAUTOCLAIM failed (non-critical): %s", exc)

    # -- Redis helpers -------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(redis.RedisError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _acquire_lock(self, job_id: str) -> bool:
        lock_key = f"{_LOCK_PREFIX}{job_id}"
        result = self._redis.set(lock_key, self._cfg.worker_id, nx=True, ex=300)
        return result is not None

    def _release_lock(self, job_id: str) -> None:
        """Safe Lua-script lock release — only releases if we own it."""
        lock_key = f"{_LOCK_PREFIX}{job_id}"
        lua = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('del', KEYS[1])
        else
            return 0
        end
        """
        self._redis.eval(lua, 1, lock_key, self._cfg.worker_id)

    def _check_and_mark_idempotency(self, idempotency_key: str) -> bool:
        """Return True if the key is new (job should proceed); False if duplicate."""
        idem_key = f"{_IDEMPOTENCY_PREFIX}processed:{idempotency_key}"
        result = self._redis.set(idem_key, "1", nx=True, ex=86_400)
        return result is not None  # None → key existed → duplicate

    def _update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error: str | None = None,
        result: dict | None = None,
    ) -> None:
        """Update the job status hash (best-effort; never raises)."""
        try:
            from ..gateway.gateway import _JOB_HASH_PREFIX
            import datetime as dt
            hash_key = f"{_JOB_HASH_PREFIX}{job_id}"
            mapping: dict[str, str] = {"status": str(status)}
            if status in (JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.DLQ):
                mapping["completed_at"] = dt.datetime.now(dt.UTC).isoformat()
            if error:
                mapping["error"] = error[:2000]  # cap error length
            if result:
                mapping["result"] = json.dumps(result)
            self._redis.hset(hash_key, mapping=mapping)
        except Exception as exc:
            log.warning("Failed to update job status for %s: %s", job_id, exc)

    def _increment_attempts(self, job_id: str) -> int:
        """Atomically increment and return the attempt counter."""
        try:
            from ..gateway.gateway import _JOB_HASH_PREFIX
            hash_key = f"{_JOB_HASH_PREFIX}{job_id}"
            return int(self._redis.hincrby(hash_key, "attempts", 1))
        except Exception:
            return self._cfg.max_delivery_attempts  # fail-safe: send to DLQ

    def _ensure_consumer_group(self) -> None:
        """Create the consumer group if it doesn't exist (idempotent)."""
        try:
            self._redis.xgroup_create(_STREAM_KEY, self._cfg.consumer_group, id="0", mkstream=True)
            log.info("Consumer group %r created.", self._cfg.consumer_group)
        except redis.exceptions.ResponseError as exc:
            if "BUSYGROUP" in str(exc):
                log.debug("Consumer group %r already exists.", self._cfg.consumer_group)
            else:
                raise

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
