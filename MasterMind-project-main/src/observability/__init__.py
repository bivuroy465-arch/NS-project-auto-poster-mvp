"""Zero-Leak OpenTelemetry Observability.

Public API
----------
    from src.observability import configure_tracing, tracer, record_exception

See telemetry.py for full documentation.
"""

from .telemetry import (
    RedactingSpanProcessor,
    configure_tracing,
    record_exception,
    tracer,
)

__all__ = [
    "RedactingSpanProcessor",
    "configure_tracing",
    "record_exception",
    "tracer",
]
