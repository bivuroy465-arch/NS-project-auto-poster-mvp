"""Self-learning feedback loop: EngagementCollector, FeedbackSubscriber, and
the sync/async boundary (`LifecycleManager`) that lets `main.py` stay fully
synchronous while this subsystem runs on its own background event loop.

Data flow (see the architecture diagram from the design discussion):

    Publishers --publish--> PostPublished
                                  |
                                  v
                        EngagementCollector (subscribes)
                          | tracks the post, polls periodically
                          v
                   EngagementSnapshotFetched
                                  |
                                  v
                        FeedbackSubscriber (subscribes)
                          | scores + records into the memory store
                          v
                 get_few_shot_exemplars(platform)
                                  |
                                  v
                content_writer.py (future integration phase)

Both `EngagementCollector` and `FeedbackSubscriber` only depend on
`event_bus.EventBus` and small `Protocol`s of their own (`EngagementClient`,
`MemoryRepository`) - neither imports the other, and neither knows anything
about `main.py`, `sheets_logger`, or `notifier`. That decoupling is the
point: swap the mock engagement client for a real Twitter/LinkedIn/Facebook
analytics client, or the in-memory repository for a Sheets/DB-backed one,
without touching this module.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import dataclasses
import datetime as dt
import random
import threading
from collections import defaultdict, deque
from collections.abc import Coroutine
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from . import sheets_logger
from .event_bus import (
    EngagementMetrics,
    EngagementSnapshotFetched,
    EventBus,
    Platform,
    PostPublished,
    Subscription,
)
from .logging_setup import get_logger

log = get_logger("feedback_loop")

__all__ = [
    "EngagementClient",
    "MockEngagementClient",
    "EngagementCollectorConfig",
    "EngagementCollector",
    "ExemplarRecord",
    "FewShotExemplar",
    "MemoryRepository",
    "InMemoryMemoryRepository",
    "SheetsMemoryRepository",
    "build_memory_repository",
    "FeedbackSubscriberConfig",
    "FeedbackSubscriber",
    "Lifecycle",
    "LifecycleManager",
    "FeedbackLoopHandle",
    "start_feedback_loop",
]

T = TypeVar("T")


# --------------------------------------------------------------------------
# Engagement collection
# --------------------------------------------------------------------------
@runtime_checkable
class EngagementClient(Protocol):
    """Strategy interface for fetching engagement from a real platform API.

    `EngagementCollector` depends only on this shape - swap
    `MockEngagementClient` for real Twitter/LinkedIn/Facebook analytics
    clients later with no change to the collector itself.
    """

    async def fetch_engagement(self, platform: Platform, post_id: str) -> EngagementMetrics: ...


class MockEngagementClient:
    """Deterministic (seedable) fake engagement source for development and tests.

    Architected purely behind `EngagementClient` so this is a drop-in
    placeholder, not a hardcoded assumption baked into the collector.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    async def fetch_engagement(self, platform: Platform, post_id: str) -> EngagementMetrics:
        await asyncio.sleep(0)  # yield control, as a real HTTP call would
        impressions = self._rng.randint(50, 5000)
        likes = self._rng.randint(0, impressions // 10 + 1)
        comments = self._rng.randint(0, likes // 5 + 1)
        shares = self._rng.randint(0, likes // 8 + 1)
        return EngagementMetrics(
            likes=likes, comments=comments, shares=shares, impressions=impressions
        )


@dataclasses.dataclass(slots=True)
class _TrackedPost:
    platform: Platform
    post_id: str
    topic: str
    text: str
    snapshots_collected: int = 0


class EngagementCollectorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    poll_interval_seconds: float = Field(default=300.0, gt=0)
    max_snapshots_per_post: int = Field(
        default=5,
        gt=0,
        description="Stop polling a post after this many snapshots (engagement plateaus).",
    )


class EngagementCollector:
    """Background periodic worker that polls engagement for tracked posts.

    Subscribes to `PostPublished` to learn what to track; on each poll,
    fetches engagement for every tracked post *concurrently* via the
    injected `EngagementClient` and publishes an `EngagementSnapshotFetched`
    per successful fetch. One post's fetch failure never blocks or drops
    the others (`asyncio.gather(..., return_exceptions=True)`).

    The periodic loop (`_run_loop`) is a thin wrapper around the directly
    callable, independently testable `poll_once()` - tests never need to
    sleep for `poll_interval_seconds` to exercise the real logic.
    """

    def __init__(
        self,
        *,
        bus: EventBus,
        client: EngagementClient,
        config: EngagementCollectorConfig | None = None,
    ) -> None:
        self._bus = bus
        self._client = client
        self._config = config or EngagementCollectorConfig()
        self._tracked: dict[tuple[str, str], _TrackedPost] = {}
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._subscription: Subscription | None = None

    @property
    def tracked_count(self) -> int:
        return len(self._tracked)

    async def start(self) -> None:
        if self._task is not None:
            return
        self._subscription = self._bus.subscribe(PostPublished, self._on_post_published)
        self._task = asyncio.create_task(self._run_loop(), name="engagement-collector")

    async def stop(self) -> None:
        if self._subscription is not None:
            self._subscription.unsubscribe()
            self._subscription = None
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _on_post_published(self, event: PostPublished) -> None:
        async with self._lock:
            key = (event.platform, event.post_id)
            self._tracked.setdefault(
                key,
                _TrackedPost(
                    platform=event.platform,
                    post_id=event.post_id,
                    topic=event.topic,
                    text=event.text,
                ),
            )

    async def _run_loop(self) -> None:
        while True:
            await asyncio.sleep(self._config.poll_interval_seconds)
            await self.poll_once()

    async def poll_once(self) -> int:
        """Fetch engagement for every tracked post once. Returns the number polled.

        Public (not just the internal loop) so callers/tests can trigger a
        deterministic poll on demand.
        """
        async with self._lock:
            snapshot = list(self._tracked.values())
        if not snapshot:
            return 0

        results = await asyncio.gather(
            *(self._client.fetch_engagement(post.platform, post.post_id) for post in snapshot),
            return_exceptions=True,
        )

        polled = 0
        for post, result in zip(snapshot, results, strict=True):
            if isinstance(result, BaseException):
                log.warning(
                    "Engagement fetch failed for %s/%s: %s", post.platform, post.post_id, result
                )
                continue
            await self._bus.publish(
                EngagementSnapshotFetched(
                    platform=post.platform,
                    post_id=post.post_id,
                    topic=post.topic,
                    text=post.text,
                    metrics=result,
                )
            )
            polled += 1
            async with self._lock:
                post.snapshots_collected += 1
                if post.snapshots_collected >= self._config.max_snapshots_per_post:
                    self._tracked.pop((post.platform, post.post_id), None)
        return polled


# --------------------------------------------------------------------------
# Feedback subscriber & prompt memory store
# --------------------------------------------------------------------------
class ExemplarRecord(BaseModel):
    """One historical (post, outcome) pair usable as a prompt exemplar."""

    model_config = ConfigDict(frozen=True)

    platform: Platform
    topic: str
    text: str
    engagement_rate: float = Field(ge=0.0)
    collected_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))


class FewShotExemplar(BaseModel):
    """What `content_writer.py` actually consumes: a labeled do-more/do-less example."""

    model_config = ConfigDict(frozen=True)

    label: Literal["best", "worst"]
    topic: str
    text: str
    engagement_rate: float


@runtime_checkable
class MemoryRepository(Protocol):
    """Durable-state Strategy: swap for a real Sheets/DB-backed store later."""

    async def save(self, record: ExemplarRecord) -> None: ...
    async def recent(self, platform: Platform, limit: int) -> list[ExemplarRecord]: ...


class InMemoryMemoryRepository:
    """Simulated durable state: a bounded, per-platform ring buffer.

    Does NOT survive a process restart. Useful as a default when Sheets
    isn't configured (e.g. local dev, `dry_run` testing) and as the
    dependency-free fake used throughout this module's own test suite.
    """

    def __init__(self, *, max_per_platform: int = 200) -> None:
        self._max_per_platform = max_per_platform
        self._by_platform: dict[str, deque[ExemplarRecord]] = defaultdict(
            lambda: deque(maxlen=max_per_platform)
        )
        self._lock = asyncio.Lock()

    async def save(self, record: ExemplarRecord) -> None:
        async with self._lock:
            self._by_platform[record.platform].append(record)

    async def recent(self, platform: Platform, limit: int) -> list[ExemplarRecord]:
        async with self._lock:
            items = list(self._by_platform.get(platform, ()))
        return items[-limit:]


def _parse_iso(value: object) -> dt.datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


class SheetsMemoryRepository:
    """Durable `MemoryRepository` backed by a dedicated Google Sheets tab.

    `sheets_logger` owns all gspread/credentials/worksheet-structure details
    (consistent with the rest of this codebase - it is the one module that
    knows how to talk to Sheets); this class is a thin async adapter that
    offloads those *blocking* HTTP calls to a worker thread via
    `asyncio.to_thread`. That is the same fix as
    `semantic_dedup.FastEmbedLocalProvider` offloading CPU-bound ONNX
    inference - same reasoning (don't block the event loop), different kind
    of blocking work (network I/O instead of CPU).

    Fails open on both read and write: `sheets_logger.log_engagement`/
    `recent_engagement` already swallow their own errors and log a warning,
    matching `sheets_logger.recent_topics()`'s "state must never break the
    pipeline" contract. A malformed row from a hand-edited sheet is skipped
    individually rather than failing the whole read.
    """

    async def save(self, record: ExemplarRecord) -> None:
        await asyncio.to_thread(
            sheets_logger.log_engagement,
            collected_at=record.collected_at.isoformat(),
            platform=record.platform,
            topic=record.topic,
            text=record.text,
            engagement_rate=record.engagement_rate,
        )

    async def recent(self, platform: Platform, limit: int) -> list[ExemplarRecord]:
        rows = await asyncio.to_thread(sheets_logger.recent_engagement, platform, limit)
        records: list[ExemplarRecord] = []
        for row in rows:
            try:
                records.append(
                    ExemplarRecord(
                        platform=row.get("platform", platform),
                        topic=row.get("topic", ""),
                        text=row.get("text", ""),
                        engagement_rate=float(row.get("engagement_rate") or 0.0),
                        collected_at=_parse_iso(row.get("collected_at")) or dt.datetime.now(dt.UTC),
                    )
                )
            except Exception as exc:
                log.warning("Skipping malformed engagement row for %s: %s", platform, exc)
        return records


def build_memory_repository() -> MemoryRepository:
    """Choose the best available `MemoryRepository`: Sheets if configured, else in-memory.

    Mirrors `sheets_logger.recent_topics()`'s own graceful-degradation
    stance: the feedback loop should work out of the box even before
    GOOGLE_SERVICE_ACCOUNT_JSON/GOOGLE_SHEET_ID are set up, it just won't
    survive a restart until they are.
    """
    if sheets_logger.is_configured():
        return SheetsMemoryRepository()
    log.warning(
        "Google Sheets is not configured (GOOGLE_SERVICE_ACCOUNT_JSON/GOOGLE_SHEET_ID); "
        "the self-learning feedback loop will use an in-memory store that does not "
        "survive restarts."
    )
    return InMemoryMemoryRepository()


class FeedbackSubscriberConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    history_window: int = Field(
        default=200,
        gt=0,
        description="How many recent posts per platform inform exemplar selection.",
    )
    exemplars_per_label: int = Field(
        default=2, gt=0, description="Default k for get_few_shot_exemplars()."
    )


class FeedbackSubscriber:
    """Consumes `EngagementSnapshotFetched` and maintains the prompt memory.

    `get_few_shot_exemplars()` is the one method `content_writer.py` will
    eventually call to bias future generations toward what has historically
    performed well (and away from what has not) - see the module docstring
    for the full data flow.

    Ranking is a plain sort by `engagement_rate` over the platform's recent
    history, not a z-score against a rolling baseline. That is a deliberate
    simplification for now, not an oversight: with the small post volumes
    an early-stage poster produces, a rolling baseline has too little data
    to be stable. It is the natural next refinement once volume justifies
    it - `get_few_shot_exemplars` is the one method that would change.
    """

    def __init__(
        self,
        *,
        bus: EventBus,
        repository: MemoryRepository | None = None,
        config: FeedbackSubscriberConfig | None = None,
    ) -> None:
        self._bus = bus
        self._repository = repository or build_memory_repository()
        self._config = config or FeedbackSubscriberConfig()
        self._subscription: Subscription | None = None

    async def start(self) -> None:
        if self._subscription is None:
            self._subscription = self._bus.subscribe(EngagementSnapshotFetched, self._on_snapshot)

    async def stop(self) -> None:
        if self._subscription is not None:
            self._subscription.unsubscribe()
            self._subscription = None

    async def _on_snapshot(self, event: EngagementSnapshotFetched) -> None:
        record = ExemplarRecord(
            platform=event.platform,
            topic=event.topic,
            text=event.text,
            engagement_rate=event.metrics.engagement_rate,
            collected_at=event.occurred_at,
        )
        await self._repository.save(record)
        log.info(
            "Recorded engagement for %s/%s: rate=%.4f",
            event.platform,
            event.post_id,
            record.engagement_rate,
        )

    async def get_few_shot_exemplars(
        self, platform: Platform, k: int | None = None
    ) -> list[FewShotExemplar]:
        """Return up to `k` best- and `k` worst-performing recent examples for `platform`.

        Returns an empty list until enough engagement history has
        accumulated - callers should treat that as "no bias available yet",
        not an error.
        """
        k = k or self._config.exemplars_per_label
        history = await self._repository.recent(platform, self._config.history_window)
        if not history:
            return []

        ranked = sorted(history, key=lambda r: r.engagement_rate, reverse=True)
        best = ranked[:k]
        # Index-based split (not value/identity based) so a small history
        # never lets the same record appear in both "best" and "worst".
        worst_start = max(k, len(ranked) - k)
        worst = ranked[worst_start:]

        return [
            FewShotExemplar(
                label="best", topic=r.topic, text=r.text, engagement_rate=r.engagement_rate
            )
            for r in best
        ] + [
            FewShotExemplar(
                label="worst", topic=r.topic, text=r.text, engagement_rate=r.engagement_rate
            )
            for r in worst
        ]


# --------------------------------------------------------------------------
# Sync/async boundary
# --------------------------------------------------------------------------
@runtime_checkable
class Lifecycle(Protocol):
    """Anything with an async start/stop can be managed by `LifecycleManager`."""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...


class LifecycleManager:
    """Owns a background thread running one persistent asyncio event loop.

    This is the sync/async boundary: `main.py`'s orchestrator stays fully
    synchronous, while the event bus, `EngagementCollector`, and
    `FeedbackSubscriber` all run on a single dedicated loop in a background
    thread - started once at process startup, stopped once at shutdown.

    A persistent loop (rather than `semantic_dedup.is_duplicate_sync`'s
    `asyncio.run()`-per-call approach) is the right call here specifically
    *because* this subsystem owns genuine background tasks of its own (the
    collector's periodic polling) that must keep running between calls from
    the synchronous side - a fresh loop per call would have nowhere to keep
    that state.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._components: list[Lifecycle] = []

    @property
    def is_running(self) -> bool:
        return self._loop is not None

    def start(self, *components: Lifecycle) -> None:
        if self._thread is not None:
            raise RuntimeError("LifecycleManager is already running")
        self._components = list(components)
        self._ready.clear()
        self._thread = threading.Thread(
            target=self._run_loop_forever, name="feedback-loop", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("Background event loop failed to start within 5 seconds")
        try:
            for component in self._components:
                self.run_coroutine(component.start()).result(timeout=10.0)
        except Exception:
            log.exception("A component failed to start; tearing down the background loop.")
            self.stop()
            raise

    def stop(self, *, timeout: float = 10.0) -> None:
        if self._loop is None or self._thread is None:
            return
        for component in reversed(self._components):
            try:
                self.run_coroutine(component.stop()).result(timeout=timeout)
            except Exception:
                log.exception("Error stopping component %r", component)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=timeout)
        self._loop = None
        self._thread = None
        self._ready = threading.Event()  # allow a future start() to wait cleanly again

    def run_coroutine(self, coro: Coroutine[Any, Any, T]) -> concurrent.futures.Future[T]:
        """Schedule `coro` onto the background loop from any (sync) thread.

        Returns a `concurrent.futures.Future`: call `.result(timeout=...)`
        to block the calling synchronous thread for the outcome, or ignore
        the return value entirely for fire-and-forget.
        """
        if self._loop is None:
            raise RuntimeError("LifecycleManager is not running")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _run_loop_forever(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()


@dataclasses.dataclass(slots=True)
class FeedbackLoopHandle:
    """Everything the synchronous orchestrator needs to talk to the feedback loop.

    This is the shape `main.py` will hold onto in the eventual integration
    phase: a couple of `..._sync` convenience methods, no `asyncio` in
    sight from the caller's side.
    """

    manager: LifecycleManager
    bus: EventBus
    collector: EngagementCollector
    subscriber: FeedbackSubscriber

    def publish_post_published_sync(self, event: PostPublished, *, timeout: float = 5.0) -> None:
        self.manager.run_coroutine(self.bus.publish(event)).result(timeout=timeout)

    def get_few_shot_exemplars_sync(
        self,
        platform: Platform,
        k: int | None = None,
        *,
        timeout: float = 5.0,
    ) -> list[FewShotExemplar]:
        return self.manager.run_coroutine(
            self.subscriber.get_few_shot_exemplars(platform, k)
        ).result(timeout=timeout)

    def shutdown(self, *, timeout: float = 10.0) -> None:
        self.manager.stop(timeout=timeout)


def start_feedback_loop(
    *,
    engagement_client: EngagementClient | None = None,
    collector_config: EngagementCollectorConfig | None = None,
    subscriber_config: FeedbackSubscriberConfig | None = None,
    repository: MemoryRepository | None = None,
) -> FeedbackLoopHandle:
    """One-call wiring for `main.py`: builds the bus and both subscribers,
    starts the background thread/loop, and returns a sync-friendly facade.

    This is the integration point for the later phase - it keeps the
    synchronous orchestrator completely unaware of asyncio.
    """
    bus = EventBus()
    collector = EngagementCollector(
        bus=bus,
        client=engagement_client or MockEngagementClient(),
        config=collector_config,
    )
    subscriber = FeedbackSubscriber(bus=bus, repository=repository, config=subscriber_config)
    manager = LifecycleManager()
    manager.start(bus, collector, subscriber)
    return FeedbackLoopHandle(manager=manager, bus=bus, collector=collector, subscriber=subscriber)
