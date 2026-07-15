"""Event Broker — Redis Streams consumer with DLQ and circuit breaker.

Public API
----------
    from src.broker import StreamConsumer, ConsumerConfig, ConsumedMessage
"""

from .consumer import (
    CircuitOpenError,
    ConsumedMessage,
    ConsumerConfig,
    StreamConsumer,
)

__all__ = [
    "CircuitOpenError",
    "ConsumedMessage",
    "ConsumerConfig",
    "StreamConsumer",
]
