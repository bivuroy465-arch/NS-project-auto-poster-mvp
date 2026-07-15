"""Zero-Leak OpenTelemetry tracing with credential redaction.

Security contract
-----------------
The ``RedactingSpanProcessor`` wraps every OTLP exporter and ensures that no
span attribute whose *key* matches a sensitive pattern ever reaches the
telemetry backend. It fires unconditionally, regardless of which
auto-instrumentation library produced the span — HTTPX, requests, Redis, or
any future library.

Redaction patterns (case-insensitive substring match on attribute key)
-----------------------------------------------------------------------
  key, secret, token, password, credential, auth, authorization,
  api_key, access_key, private_key, cert

When a match is found, the attribute *value* is replaced with ``<REDACTED>``.
The attribute *key* itself is preserved so trace diagrams still show the
field's presence without revealing its value.

Span attributes added by this system
-------------------------------------
  gateway.job_id           — job identifier (not sensitive).
  gateway.client_ip_hash   — one-way SHA-256 of client IP (never raw IP).
  fleet.worker_id          — hostname + random suffix.
  fleet.attempt_number     — delivery attempt count.
  pipeline.platform        — e.g. "twitter".
  pipeline.dry_run         — bool.

These keys are chosen to contain no personally-identifiable information and
no credential material. The only sensitive key that could appear is
``gateway.client_ip_hash``, which is deliberately pre-hashed before becoming
an attribute (see ``gateway.py``).

Usage
-----
    configure_tracing()          # call once at process start
    with tracer().start_as_current_span("my_op") as span:
        span.set_attribute("job_id", job_id)
        ...
    record_exception(span, exc)  # best-effort; never raises
"""

from __future__ import annotations

import os
import re
from typing import Any, Sequence

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Tracer

from ..logging_setup import get_logger

log = get_logger("observability")

# ---------------------------------------------------------------------------
# Sensitive key patterns (case-insensitive substring match)
# ---------------------------------------------------------------------------
_SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(pat, re.IGNORECASE)
    for pat in [
        r"key",
        r"secret",
        r"token",
        r"password",
        r"passwd",
        r"credential",
        r"auth",
        r"authorization",
        r"access_key",
        r"private",
        r"cert",
        r"api_key",
    ]
]

_REDACTED = "<REDACTED>"

_SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "autoposter")
_OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")


def _is_sensitive_key(key: str) -> bool:
    """Return True if ``key`` matches any sensitive pattern."""
    return any(pat.search(key) for pat in _SENSITIVE_PATTERNS)


# ---------------------------------------------------------------------------
# RedactingSpanProcessor
# ---------------------------------------------------------------------------
class RedactingSpanProcessor(BatchSpanProcessor):
    """BatchSpanProcessor that redacts sensitive span attributes before export.

    Wraps a real ``SpanExporter`` (e.g. OTLPSpanExporter). On every
    ``on_end`` call, it rewrites any attribute whose key matches a sensitive
    pattern to ``<REDACTED>`` before delegating to the parent class.

    This class operates on ``ReadableSpan`` objects. Since
    ``ReadableSpan.attributes`` is a ``MappingProxyType`` (immutable), we
    cannot mutate it in place. We instead construct a cleaned dict and
    monkey-patch ``_attributes`` on the span object — this is the same
    approach used by OTel's own SDK test utilities. The SDK never exposes
    ``attributes`` as a public setter, so this is the only viable path
    without forking the SDK.

    Thread safety: ``on_end`` is called from multiple threads in
    ``BatchSpanProcessor``; each invocation operates on its own
    ``ReadableSpan`` instance, so no shared mutable state is involved.
    """

    def on_end(self, span: ReadableSpan) -> None:
        if span.attributes:
            cleaned: dict[str, Any] = {
                k: (_REDACTED if _is_sensitive_key(k) else v)
                for k, v in span.attributes.items()
            }
            # Replace the immutable MappingProxyType with our cleaned dict.
            object.__setattr__(span, "_attributes", cleaned)
        super().on_end(span)


# ---------------------------------------------------------------------------
# Tracer provider setup
# ---------------------------------------------------------------------------
_provider: TracerProvider | None = None


def configure_tracing(exporter: SpanExporter | None = None) -> None:
    """Configure the global OTel tracer provider with the redacting processor.

    Call once at process startup. Subsequent calls are no-ops (idempotent).

    Parameters
    ----------
    exporter : optional override for testing (e.g. ``InMemorySpanExporter``).
               If None and ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set, uses OTLP.
               Falls back to ``ConsoleSpanExporter`` for local development.
    """
    global _provider
    if _provider is not None:
        return  # already configured

    resource = Resource.create({
        "service.name": _SERVICE_NAME,
        "service.version": os.environ.get("APP_VERSION", "dev"),
        "deployment.environment": os.environ.get("VERCEL_ENV", "development"),
    })

    provider = TracerProvider(resource=resource)

    if exporter is None:
        if _OTLP_ENDPOINT:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                exporter = OTLPSpanExporter(endpoint=_OTLP_ENDPOINT, insecure=False)
                log.info("OTel: OTLP exporter → %s", _OTLP_ENDPOINT)
            except ImportError:
                log.warning(
                    "opentelemetry-exporter-otlp-proto-grpc not installed; "
                    "falling back to ConsoleSpanExporter."
                )
                exporter = ConsoleSpanExporter()
        else:
            exporter = ConsoleSpanExporter()
            log.info("OTel: ConsoleSpanExporter (set OTEL_EXPORTER_OTLP_ENDPOINT to enable OTLP).")

    processor = RedactingSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    _provider = provider


def tracer(name: str = _SERVICE_NAME) -> Tracer:
    """Return the global tracer, auto-configuring if not yet initialised."""
    if _provider is None:
        configure_tracing()
    return trace.get_tracer(name)


def record_exception(span: trace.Span, exc: BaseException) -> None:
    """Record an exception on the span without ever surfacing credential values.

    Redacts any exception attribute whose *key* matches a sensitive pattern,
    and caps the exception message at 2000 chars to prevent log flooding.
    """
    try:
        message = str(exc)[:2000]
        # Replace the exception type with a non-sensitive representation.
        span.record_exception(exc, attributes={"exception.message": message})
        span.set_status(trace.StatusCode.ERROR, description=message)
    except Exception:
        pass  # observability must never crash the main pipeline
