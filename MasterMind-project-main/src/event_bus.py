"""In-process async event bus with a durable, append-only event log.

This is the foundational module for the self-learning feedback loop (and the
canonical home for `DomainEvent`, also used by `semantic_dedup.py`). Two
things are combined deliberately, not accidentally:

    1. An `EventStore` (append-only, durable-shaped) that records every
       event *before* anything reacts to it - the actual "event sourcing"
       part. `InMemoryEventStore` stands in for a real store (Sheets, a
       small SQLite table, ...) via the same `EventStore` Protocol.
    2. A live, best-effort dispatch queue that fans events out to
       subscribers in the background.

These two concerns are intentionally decoupled: `publish()` always appends
to the store first (so the event is durably on record even if the process
dies a microsecond later), then tries to enqueue it for live dispatch. If
the live queue is under pressure (a slow/stuck subscriber), the default
`OverflowPolicy.DROP_OLDEST` policy drops the oldest *still-queued* event
from *live delivery* only - it is never lost from the store. This keeps
`publish()` itself always fast and non-blocking for hot paths (e.g.
`SemanticDeduplicator.check()`), matching this codebase's established
"observability must never slow down or break the main pipeline" philosophy.

Subscriptions are type-based and support inheritance: subscribing to
`DomainEvent` itself receives every event (useful for a generic logger or
metrics sink); subscribing to a specific subclass receives only that type
(and its subclasses).

Concurrency notes:
    - `_registrations` is guarded by a `threading.Lock`, not `asyncio.Lock`:
      `subscribe()`/`unsubscribe()` are plain synchronous methods with no
      `await` inside, so a lightweight OS-level lock is both correct and
      cheap regardless of which thread calls them (relevant once
      `feedback_loop.LifecycleManager` starts calling into this bus from a
      background thread).
    - The dispatch loop is a single `asyncio.Task` reading off one
      `asyncio.Queue`, so handler invocation for a given event is
      sequenced, but handlers for the *same* event fan out concurrently via
      `asyncio.gather` with `return_exceptions=True` - one broken
      subscriber can never break its siblings or the loop itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime as dt
import threading
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from enum import StrEnum
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from .logging_setup import get_logger

log = get_logger("event_bus")

__all__ = [
    "Platform",
    "DomainEvent",
    "EngagementMetrics",
    "PostPublished",
    "EngagementSnapshotFetched",
    "EventStore",
    "InMemoryEventStore",
    "OverflowPolicy",
    "EventBusConfig",
    "Subscription",
    "EventBus",
]

# Mirrors `config_loader._VALID_PLATFORMS`; duplicated as a small literal
# here rather than imported, since that name is private to config_loader
# and this module intentionally has no dependency on it.
Platform = Literal["twitter", "linkedin", "facebook"]

TEvent = TypeVar("TEvent", bound="DomainEvent")
Handler = Callable[[TEvent], Awaitable[None]]


# --------------------------------------------------------------------------
# Domain events (strict pydantic v2 schemas)
# --------------------------------------------------------------------------
class DomainEvent(BaseModel):
    """Base type for every event flowing through the bus.

    Every event carries its own identity and timestamp, independent of
    transport - the hallmark of an event-sourcing schema, as opposed to a
    plain callback payload.
    """

    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    occurred_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))


class EngagementMetrics(BaseModel):
    """A single point-in-time engagement reading for a published post."""

    model_config = ConfigDict(frozen=True)

    likes: int = Field(default=0, ge=0)
    comments: int = Field(default=0, ge=0)
    shares: int = Field(default=0, ge=0)
    impressions: int = Field(default=0, ge=0)

    @property
    def engagement_rate(self) -> float:
        """(likes + comments + shares) / impressions, or 0.0 with no impressions."""
        if self.impressions == 0:
            return 0.0
        return (self.likes + self.comments + self.shares) / self.impressions


class PostPublished(DomainEvent):
    """A post was successfully published to a platform.

    Published by the orchestrator (eventually `main.py`/`publishers.*`)
    right after a successful publish call - this is what seeds the
    `EngagementCollector`'s tracking list.
    """

    platform: Platform
    post_id: str
    topic: str
    text: str
    image_url: str = ""


class EngagementSnapshotFetched(DomainEvent):
    """A periodic engagement reading for a previously published post."""

    platform: Platform
    post_id: str
    topic: str
    text: str
    metrics: EngagementMetrics


# --------------------------------------------------------------------------
# Durable event log (the "event sourcing" half of this module)
# --------------------------------------------------------------------------
@runtime_checkable
class EventStore(Protocol):
    """Strategy interface for the durable, append-only event log."""

    async def append(self, event: DomainEvent) -> None: ...
    async def all_events(self) -> tuple[DomainEvent, ...]: ...
    async def events_of_type(self, event_type: type[TEvent]) -> tuple[TEvent, ...]: ...


class InMemoryEventStore:
    """Bounded in-memory event log.

    Stands in for a real durable store (a dedicated Sheets tab, SQLite,
    etc.) behind the `EventStore` Protocol. Bounded via `maxlen` so a
    long-running process cannot grow this unboundedly; swap for a real
    store before relying on it as a genuine audit trail.
    """

    def __init__(self, *, max_events: int | None = 10_000) -> None:
        self._events: deque[DomainEvent] = deque(maxlen=max_events)
        self._lock = asyncio.Lock()

    async def append(self, event: DomainEvent) -> None:
        async with self._lock:
            self._events.append(event)

    async def all_events(self) -> tuple[DomainEvent, ...]:
        async with self._lock:
            return tuple(self._events)

    async def events_of_type(self, event_type: type[TEvent]) -> tuple[TEvent, ...]:
        async with self._lock:
            return tuple(e for e in self._events if isinstance(e, event_type))


# --------------------------------------------------------------------------
# Live dispatch (the "pub/sub" half of this module)
# --------------------------------------------------------------------------
class OverflowPolicy(StrEnum):
    """What to do when the live-dispatch queue is full.

    Either way, the event is never lost from the `EventStore` - only from
    *live* delivery.
    """

    DROP_OLDEST = "drop-oldest"  # stay responsive; lose the oldest queued live-notification
    BLOCK = "block"  # apply true backpressure to publishers instead


class EventBusConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_queue_size: int = Field(default=1000, gt=0)
    overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST


@dataclasses.dataclass(slots=True)
class _Registration:
    event_type: type[DomainEvent]
    handler: Callable[[Any], Awaitable[None]]


class Subscription:
    """Disposable handle returned by `EventBus.subscribe()`."""

    __slots__ = ("_bus", "_registration", "_active")

    def __init__(self, bus: EventBus, registration: _Registration) -> None:
        self._bus = bus
        self._registration = registration
        self._active = True

    def unsubscribe(self) -> None:
        if self._active:
            self._bus._remove(self._registration)
            self._active = False


class EventBus:
    """The async event bus: durable log + best-effort live dispatch.

    Lifecycle: construct, `await start()` to begin dispatching, `subscribe()`
    any time (before or after `start()`), `await publish(...)` to emit
    events, `await stop()` when done. Also usable as an async context
    manager (`async with EventBus() as bus: ...`).
    """

    def __init__(
        self, *, config: EventBusConfig | None = None, store: EventStore | None = None
    ) -> None:
        self._config = config or EventBusConfig()
        self._store = store or InMemoryEventStore()
        self._queue: asyncio.Queue[DomainEvent] = asyncio.Queue(maxsize=self._config.max_queue_size)
        self._registrations: list[_Registration] = []
        self._sub_lock = threading.Lock()  # see module docstring: sync, cross-thread safe
        self._dispatch_task: asyncio.Task[None] | None = None

    @property
    def store(self) -> EventStore:
        return self._store

    def subscribe(self, event_type: type[TEvent], handler: Handler[TEvent]) -> Subscription:
        """Register `handler` for `event_type` and all of its subclasses.

        Subscribing to `DomainEvent` itself is a valid wildcard: the handler
        receives every event published on this bus.
        """
        registration = _Registration(event_type=event_type, handler=handler)
        with self._sub_lock:
            self._registrations.append(registration)
        return Subscription(self, registration)

    def _remove(self, registration: _Registration) -> None:
        with self._sub_lock:
            if registration in self._registrations:
                self._registrations.remove(registration)

    async def publish(self, event: DomainEvent) -> None:
        """Durably record `event`, then best-effort enqueue it for live dispatch."""
        await self._store.append(event)
        await self._enqueue(event)

    async def _enqueue(self, event: DomainEvent) -> None:
        if self._config.overflow_policy is OverflowPolicy.BLOCK:
            await self._queue.put(event)
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            dropped = self._queue.get_nowait()
            self._queue.task_done()
            log.warning(
                "Event bus live queue full (max=%d); dropped oldest queued %s from live "
                "dispatch (it remains recorded in the event store).",
                self._config.max_queue_size,
                type(dropped).__name__,
            )
            self._queue.put_nowait(event)

    async def flush(self) -> None:
        """Block until every currently-enqueued event has been dispatched."""
        await self._queue.join()

    async def start(self) -> None:
        if self._dispatch_task is None:
            self._dispatch_task = asyncio.create_task(
                self._dispatch_loop(), name="event-bus-dispatch"
            )

    async def stop(self, *, drain: bool = True) -> None:
        task = self._dispatch_task
        if task is None:
            return
        if drain:
            await self.flush()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self._dispatch_task = None

    async def _dispatch_loop(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                await self._dispatch(event)
            finally:
                self._queue.task_done()

    async def _dispatch(self, event: DomainEvent) -> None:
        with self._sub_lock:
            matched: Sequence[Callable[[Any], Awaitable[None]]] = [
                reg.handler for reg in self._registrations if isinstance(event, reg.event_type)
            ]
        if not matched:
            return
        results = await asyncio.gather(
            *(handler(event) for handler in matched), return_exceptions=True
        )
        for handler, result in zip(matched, results, strict=True):
            if isinstance(result, BaseException):
                log.error(
                    "Event handler %r raised while handling %s: %s",
                    getattr(handler, "__qualname__", handler),
                    type(event).__name__,
                    result,
                )

    async def __aenter__(self) -> EventBus:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()
