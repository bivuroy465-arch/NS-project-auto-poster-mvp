"""Processing Fleet — distributed Redis Stream consumer with idempotency locks.

Public API
----------
    from src.fleet import Worker, WorkerConfig, run_worker
    from src.fleet import DistributedLock, LockConfig, LockHandle, LockNotAcquiredError

See worker.py and lock.py for full documentation.
"""

from .lock import (
    DistributedLock,
    LockConfig,
    LockHandle,
    LockNotAcquiredError,
)
from .worker import (
    DLQError,
    JobMessage,
    Worker,
    WorkerConfig,
    run_worker,
)

__all__ = [
    "DLQError",
    "DistributedLock",
    "JobMessage",
    "LockConfig",
    "LockHandle",
    "LockNotAcquiredError",
    "Worker",
    "WorkerConfig",
    "run_worker",
]
