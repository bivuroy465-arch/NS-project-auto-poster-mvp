"""Edge Gateway — JWT auth, rate limiting, and Redis Stream job broker.

Public API
----------
    from src.gateway import GatewayRouter, enqueue_job, get_job_status

See gateway.py for full documentation.
"""

from .gateway import (
    EnqueueRequest,
    EnqueueResponse,
    GatewayConfig,
    GatewayRouter,
    JobStatus,
    JobStatusResponse,
    RateLimitExceededError,
    RedisJobBroker,
    TokenInvalidError,
    enqueue_job,
    get_job_status,
)

__all__ = [
    "EnqueueRequest",
    "EnqueueResponse",
    "GatewayConfig",
    "GatewayRouter",
    "JobStatus",
    "JobStatusResponse",
    "RateLimitExceededError",
    "RedisJobBroker",
    "TokenInvalidError",
    "enqueue_job",
    "get_job_status",
]
