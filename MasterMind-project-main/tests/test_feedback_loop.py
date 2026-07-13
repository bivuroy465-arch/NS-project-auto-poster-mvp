"""Tests for EngagementCollector, FeedbackSubscriber, and the LifecycleManager
sync/async bridge.
"""

import asyncio

import pytest

from src import sheets_logger
from src.event_bus import EngagementMetrics, EngagementSnapshotFetched, EventBus, PostPublished
from src.feedback_loop import (
    EngagementCollector,
    EngagementCollectorConfig,
    ExemplarRecord,
    FeedbackSubscriber,
    FeedbackSubscriberConfig,
    InMemoryMemoryRepository,
    LifecycleManager,
    MockEngagementClient,
    SheetsMemoryRepository,
    build_memory_repository,
    start_feedback_loop,
)


@pytest.fixture(autouse=True)
def _no_ambient_sheets_credentials(monkeypatch):
    """Guarantee these tests never depend on (or accidentally hit) real
    Google Sheets, regardless of what's set in the host environment.
    """
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.delenv("GOOGLE_SHEET_ID", raising=False)


class _FlakyClient:
    """Fails for specific post_ids, succeeds deterministically otherwise."""

    def __init__(self, fail_post_ids):
        self._fail_post_ids = set(fail_post_ids)

    async def fetch_engagement(self, platform, post_id):
        if post_id in self._fail_post_ids:
            raise RuntimeError("api down")
        return EngagementMetrics(likes=10, comments=1, shares=1, impressions=100)


class _FailingComponent:
    async def start(self):
        raise RuntimeError("cannot start")

    async def stop(self):
        return None


# --------------------------------------------------------------------------
# EngagementCollector
# --------------------------------------------------------------------------
def test_collector_tracks_and_polls_published_posts():
    bus = EventBus()
    collector = EngagementCollector(bus=bus, client=MockEngagementClient(seed=42))
    received = []

    async def handler(event):
        received.append(event)

    async def _run():
        bus.subscribe(EngagementSnapshotFetched, handler)
        await bus.start()
        await collector.start()
        await bus.publish(
            PostPublished(platform="twitter", post_id="p1", topic="AI", text="hello #ai")
        )
        await bus.flush()
        assert collector.tracked_count == 1
        polled = await collector.poll_once()
        await bus.flush()
        await collector.stop()
        await bus.stop()
        return polled

    polled = asyncio.run(_run())
    assert polled == 1
    assert len(received) == 1
    assert received[0].post_id == "p1"
    assert 0.0 <= received[0].metrics.engagement_rate <= 1.0


def test_collector_stops_tracking_after_max_snapshots():
    bus = EventBus()
    config = EngagementCollectorConfig(max_snapshots_per_post=2, poll_interval_seconds=9999)
    collector = EngagementCollector(bus=bus, client=MockEngagementClient(seed=1), config=config)

    async def _run():
        await bus.start()
        await collector.start()
        await bus.publish(PostPublished(platform="twitter", post_id="p1", topic="AI", text="hi"))
        await bus.flush()
        await collector.poll_once()
        await collector.poll_once()
        assert collector.tracked_count == 0  # evicted after max_snapshots_per_post polls
        polled_again = await collector.poll_once()
        await collector.stop()
        await bus.stop()
        return polled_again

    polled_again = asyncio.run(_run())
    assert polled_again == 0


def test_collector_isolates_per_post_failures():
    bus = EventBus()
    client = _FlakyClient(fail_post_ids={"bad-1"})
    collector = EngagementCollector(bus=bus, client=client)
    received = []

    async def handler(event):
        received.append(event)

    async def _run():
        bus.subscribe(EngagementSnapshotFetched, handler)
        await bus.start()
        await collector.start()
        await bus.publish(PostPublished(platform="twitter", post_id="bad-1", topic="t1", text="x1"))
        await bus.publish(
            PostPublished(platform="twitter", post_id="good-1", topic="t2", text="x2")
        )
        await bus.flush()
        polled = await collector.poll_once()
        await bus.flush()
        await collector.stop()
        await bus.stop()
        return polled

    polled = asyncio.run(_run())
    assert polled == 1  # only the good post produced a snapshot
    assert len(received) == 1
    assert received[0].post_id == "good-1"


# --------------------------------------------------------------------------
# FeedbackSubscriber
# --------------------------------------------------------------------------
def test_feedback_subscriber_records_snapshots():
    bus = EventBus()
    repository = InMemoryMemoryRepository()
    subscriber = FeedbackSubscriber(bus=bus, repository=repository)

    async def _run():
        await bus.start()
        await subscriber.start()
        await bus.publish(
            EngagementSnapshotFetched(
                platform="twitter",
                post_id="p1",
                topic="AI",
                text="hi #ai",
                metrics=EngagementMetrics(likes=10, comments=0, shares=0, impressions=100),
            )
        )
        await bus.flush()
        await subscriber.stop()
        await bus.stop()
        return await repository.recent("twitter", 10)

    records = asyncio.run(_run())
    assert len(records) == 1
    assert records[0].engagement_rate == pytest.approx(0.1)


def test_get_few_shot_exemplars_returns_best_and_worst():
    bus = EventBus()
    subscriber = FeedbackSubscriber(
        bus=bus,
        repository=InMemoryMemoryRepository(),
        config=FeedbackSubscriberConfig(exemplars_per_label=1),
    )

    async def _run():
        await bus.start()
        await subscriber.start()
        rates = [0.9, 0.1, 0.5, 0.05, 0.8]
        for i, rate in enumerate(rates):
            await bus.publish(
                EngagementSnapshotFetched(
                    platform="twitter",
                    post_id=f"p{i}",
                    topic=f"topic-{i}",
                    text=f"text-{i}",
                    metrics=EngagementMetrics(
                        likes=int(rate * 100), comments=0, shares=0, impressions=100
                    ),
                )
            )
        await bus.flush()
        exemplars = await subscriber.get_few_shot_exemplars("twitter")
        await subscriber.stop()
        await bus.stop()
        return exemplars

    exemplars = asyncio.run(_run())
    best = [e for e in exemplars if e.label == "best"]
    worst = [e for e in exemplars if e.label == "worst"]
    assert len(best) == 1 and best[0].topic == "topic-0"  # rate 0.9, highest
    assert len(worst) == 1 and worst[0].topic == "topic-3"  # rate 0.05, lowest


def test_get_few_shot_exemplars_empty_without_history():
    bus = EventBus()
    subscriber = FeedbackSubscriber(bus=bus, repository=InMemoryMemoryRepository())

    exemplars = asyncio.run(subscriber.get_few_shot_exemplars("twitter"))
    assert exemplars == []


def test_get_few_shot_exemplars_no_overlap_with_small_history():
    bus = EventBus()
    subscriber = FeedbackSubscriber(
        bus=bus,
        repository=InMemoryMemoryRepository(),
        config=FeedbackSubscriberConfig(exemplars_per_label=2),
    )

    async def _run():
        await bus.start()
        await subscriber.start()
        await bus.publish(
            EngagementSnapshotFetched(
                platform="twitter",
                post_id="p1",
                topic="only-one",
                text="x",
                metrics=EngagementMetrics(likes=5, comments=0, shares=0, impressions=100),
            )
        )
        await bus.flush()
        exemplars = await subscriber.get_few_shot_exemplars("twitter")
        await subscriber.stop()
        await bus.stop()
        return exemplars

    exemplars = asyncio.run(_run())
    assert len(exemplars) == 1
    assert exemplars[0].label == "best"


# --------------------------------------------------------------------------
# LifecycleManager (the sync/async boundary)
# --------------------------------------------------------------------------
def test_lifecycle_manager_rejects_double_start():
    manager = LifecycleManager()
    bus = EventBus()
    manager.start(bus)
    try:
        with pytest.raises(RuntimeError):
            manager.start(bus)
    finally:
        manager.stop()


def test_lifecycle_manager_run_coroutine_before_start_raises():
    manager = LifecycleManager()
    bus = EventBus()
    coro = bus.publish(PostPublished(platform="twitter", post_id="1", topic="t", text="x"))
    try:
        with pytest.raises(RuntimeError):
            manager.run_coroutine(coro)
    finally:
        coro.close()  # never scheduled: avoid a "coroutine was never awaited" warning


def test_lifecycle_manager_tears_down_cleanly_on_component_start_failure():
    manager = LifecycleManager()
    with pytest.raises(RuntimeError, match="cannot start"):
        manager.start(_FailingComponent())
    assert manager.is_running is False

    # Must be startable again afterwards - a failed start should never wedge the thread.
    bus = EventBus()
    manager.start(bus)
    manager.stop()


# --------------------------------------------------------------------------
# Sheets-backed durable state
# --------------------------------------------------------------------------
def test_sheets_memory_repository_save_delegates_to_sheets_logger(monkeypatch):
    calls = []
    monkeypatch.setattr(sheets_logger, "log_engagement", lambda **kwargs: calls.append(kwargs))
    repository = SheetsMemoryRepository()
    record = ExemplarRecord(platform="twitter", topic="AI", text="hi #ai", engagement_rate=0.5)

    asyncio.run(repository.save(record))

    assert len(calls) == 1
    assert calls[0]["platform"] == "twitter"
    assert calls[0]["engagement_rate"] == 0.5


def test_sheets_memory_repository_recent_delegates_and_parses_rows(monkeypatch):
    def _fake_recent_engagement(platform, limit):
        return [
            {
                "collected_at": "2026-01-01T00:00:00+00:00",
                "platform": platform,
                "topic": "AI",
                "text": "hi #ai",
                "engagement_rate": 0.5,
            }
        ]

    monkeypatch.setattr(sheets_logger, "recent_engagement", _fake_recent_engagement)
    repository = SheetsMemoryRepository()

    records = asyncio.run(repository.recent("twitter", 10))

    assert len(records) == 1
    assert records[0].topic == "AI"
    assert records[0].engagement_rate == 0.5


def test_sheets_memory_repository_recent_skips_malformed_rows(monkeypatch):
    def _fake_recent_engagement(platform, limit):
        return [
            {"platform": platform, "topic": "bad", "text": "x", "engagement_rate": "not-a-number"},
            {"platform": platform, "topic": "good", "text": "y", "engagement_rate": 0.5},
        ]

    monkeypatch.setattr(sheets_logger, "recent_engagement", _fake_recent_engagement)
    repository = SheetsMemoryRepository()

    records = asyncio.run(repository.recent("twitter", 10))

    assert len(records) == 1
    assert records[0].topic == "good"


def test_build_memory_repository_uses_sheets_when_configured(monkeypatch):
    monkeypatch.setattr(sheets_logger, "is_configured", lambda: True)
    assert isinstance(build_memory_repository(), SheetsMemoryRepository)


def test_build_memory_repository_falls_back_to_in_memory_otherwise(monkeypatch):
    monkeypatch.setattr(sheets_logger, "is_configured", lambda: False)
    assert isinstance(build_memory_repository(), InMemoryMemoryRepository)


def test_lifecycle_manager_runs_full_pipeline_from_sync_code():
    """Capstone test: drive the entire feedback loop from plain sync code,
    exactly how main.py will eventually use it - no asyncio.run() here.
    """
    handle = start_feedback_loop(
        engagement_client=MockEngagementClient(seed=7),
        repository=InMemoryMemoryRepository(),
    )
    try:
        handle.publish_post_published_sync(
            PostPublished(platform="twitter", post_id="p1", topic="Edge AI", text="hello #ai")
        )
        handle.manager.run_coroutine(handle.bus.flush()).result(timeout=5.0)
        assert handle.collector.tracked_count == 1

        polled = handle.manager.run_coroutine(handle.collector.poll_once()).result(timeout=5.0)
        assert polled == 1
        handle.manager.run_coroutine(handle.bus.flush()).result(timeout=5.0)

        exemplars = handle.get_few_shot_exemplars_sync("twitter")
        assert len(exemplars) == 1
        assert exemplars[0].topic == "Edge AI"
    finally:
        handle.shutdown()
