"""Processing Fleet — distributed Redis Stream consumer with idempotency locks.

Public API
----------
    from src.fleet import Worker, WorkerConfig, run_worker

See worker.py for full documentation.
"""

from .worker import (
    DLQError,
    JobMessage,
    Worker,
    WorkerConfig,
    run_worker,
)

__all__ = [
    "DLQError",
    "JobMessage",
    "Worker",
    "WorkerConfig",
    "run_worker",
]
