"""Tests for the async event bus: durable log + best-effort live dispatch."""

import asyncio
from uuid import UUID

import pytest
from pydantic import ValidationError

from src.event_bus import (
    DomainEvent,
    EngagementMetrics,
    EventBus,
    EventBusConfig,
    InMemoryEventStore,
    OverflowPolicy,
    PostPublished,
)


class _DummyEvent(DomainEvent):
    n: int


def test_subscribe_and_publish_delivers_event():
    bus = EventBus()
    received = []

    async def handler(event):
        received.append(event)

    async def _run():
        bus.subscribe(_DummyEvent, handler)
        await bus.start()
        await bus.publish(_DummyEvent(n=1))
        await bus.flush()
        await bus.stop()

    asyncio.run(_run())
    assert len(received) == 1
    assert received[0].n == 1
    assert isinstance(received[0].event_id, UUID)


def test_wildcard_subscription_via_base_class_receives_everything():
    bus = EventBus()
    received = []

    async def handler(event):
        received.append(event)

    async def _run():
        bus.subscribe(DomainEvent, handler)  # wildcard
        await bus.start()
        await bus.publish(_DummyEvent(n=1))
        await bus.flush()
        await bus.stop()

    asyncio.run(_run())
    assert len(received) == 1
    assert isinstance(received[0], _DummyEvent)


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    received = []

    async def handler(event):
        received.append(event)

    async def _run():
        subscription = bus.subscribe(_DummyEvent, handler)
        await bus.start()
        await bus.publish(_DummyEvent(n=1))
        await bus.flush()
        subscription.unsubscribe()
        await bus.publish(_DummyEvent(n=2))
        await bus.flush()
        await bus.stop()

    asyncio.run(_run())
    assert len(received) == 1


def test_handler_exception_does_not_break_dispatch_or_siblings():
    bus = EventBus()
    received = []

    async def bad_handler(event):
        raise RuntimeError("boom")

    async def good_handler(event):
        received.append(event)

    async def _run():
        bus.subscribe(_DummyEvent, bad_handler)
        bus.subscribe(_DummyEvent, good_handler)
        await bus.start()
        await bus.publish(_DummyEvent(n=1))
        await bus.flush()
        await bus.publish(_DummyEvent(n=2))  # bus must still work after a handler blew up
        await bus.flush()
        await bus.stop()

    asyncio.run(_run())
    assert len(received) == 2


def test_overflow_drop_oldest_still_records_everything_in_the_store():
    config = EventBusConfig(max_queue_size=2, overflow_policy=OverflowPolicy.DROP_OLDEST)
    bus = EventBus(config=config)

    async def _run():
        # Deliberately never start() the dispatch loop, so the live queue
        # fills past capacity and the overflow policy must kick in.
        for i in range(5):
            await bus.publish(_DummyEvent(n=i))
        return await bus.store.all_events()

    events = asyncio.run(_run())
    assert len(events) == 5  # durability: all 5 recorded regardless of live-queue capacity
    assert [e.n for e in events] == [0, 1, 2, 3, 4]


def test_async_context_manager_starts_and_stops():
    bus = EventBus()
    received = []

    async def handler(event):
        received.append(event)

    async def _run():
        bus.subscribe(_DummyEvent, handler)
        async with bus:
            await bus.publish(_DummyEvent(n=1))
            await bus.flush()

    asyncio.run(_run())
    assert len(received) == 1


def test_in_memory_event_store_filters_by_type():
    store = InMemoryEventStore()

    async def _run():
        await store.append(_DummyEvent(n=1))
        await store.append(PostPublished(platform="twitter", post_id="1", topic="t", text="x"))
        return await store.events_of_type(PostPublished)

    result = asyncio.run(_run())
    assert len(result) == 1
    assert isinstance(result[0], PostPublished)


def test_engagement_metrics_rate_computation():
    metrics = EngagementMetrics(likes=10, comments=5, shares=5, impressions=200)
    assert metrics.engagement_rate == pytest.approx(0.1)


def test_engagement_metrics_zero_impressions_is_zero_rate():
    assert EngagementMetrics().engagement_rate == 0.0


def test_post_published_rejects_unknown_platform():
    with pytest.raises(ValidationError):
        PostPublished(platform="mastodon", post_id="1", topic="t", text="x")
