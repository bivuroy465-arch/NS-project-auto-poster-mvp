"""Application-level composition root for the async subsystems.

Bundles the shared background event loop (`LifecycleManager`), the semantic
duplicate guard, and the self-learning feedback loop into one object with a
synchronous facade (`AppRuntime`), so `main.py` stays a plain, readable,
fully synchronous pipeline while the components underneath are async-native.

Deliberately ONE `LifecycleManager` (one background loop) for the whole
application, not one per subsystem: `SemanticDeduplicator.check()`/
`.hydrate()` and the feedback loop's `EngagementCollector`/
`FeedbackSubscriber` all run on the same loop, submitted through the same
`LifecycleManager.run_coroutine()` bridge. That is strictly better here than
`semantic_dedup.is_duplicate_sync`'s `asyncio.run()`-per-call approach: this
process already needs a persistent loop for the feedback loop's background
polling, so reusing it for dedup checks avoids spinning up a second thread
and a second loop for no reason. `is_duplicate_sync` remains the right tool
for callers that have no persistent loop available at all (a one-off script
or test).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Coroutine, Sequence
from typing import Any, TypeVar

from .config_loader import env
from .event_bus import EventBus, Platform, PostPublished
from .exceptions import AutoPosterError
from .feedback_loop import (
    EngagementCollector,
    EngagementCollectorConfig,
    FeedbackSubscriber,
    FeedbackSubscriberConfig,
    FewShotExemplar,
    LifecycleManager,
    MockEngagementClient,
    build_memory_repository,
)
from .logging_setup import get_logger
from .semantic_dedup import (
    DedupVerdict,
    SemanticDedupConfig,
    SemanticDeduplicator,
    create_deduplicator,
)

log = get_logger("runtime")

__all__ = ["AppRuntime", "start_runtime"]

T = TypeVar("T")


@dataclasses.dataclass(slots=True)
class AppRuntime:
    """Synchronous facade over every async subsystem `main.py` needs.

    Mirrors `feedback_loop.FeedbackLoopHandle`, extended to also cover the
    semantic duplicate guard on the same shared loop. `main.py` never
    imports `asyncio`; every method here is a plain, blocking call backed
    by a background event loop under the hood.
    """

    manager: LifecycleManager
    bus: EventBus
    deduplicator: SemanticDeduplicator
    collector: EngagementCollector
    subscriber: FeedbackSubscriber

    def check_topic_sync(self, topic: str, *, timeout: float = 20.0) -> DedupVerdict:
        return self._run(self.deduplicator.check(topic), timeout=timeout)

    def hydrate_topics_sync(self, recent_topics: Sequence[str], *, timeout: float = 30.0) -> None:
        self._run(self.deduplicator.hydrate(recent_topics), timeout=timeout)

    def publish_post_published_sync(self, event: PostPublished, *, timeout: float = 5.0) -> None:
        self._run(self.bus.publish(event), timeout=timeout)

    def get_few_shot_exemplars_sync(
        self,
        platform: Platform,
        k: int | None = None,
        *,
        timeout: float = 5.0,
    ) -> list[FewShotExemplar]:
        return self._run(self.subscriber.get_few_shot_exemplars(platform, k), timeout=timeout)

    def shutdown(self, *, timeout: float = 10.0) -> None:
        self.manager.stop(timeout=timeout)

    def _run(self, coro: Coroutine[Any, Any, T], *, timeout: float) -> T:
        return self.manager.run_coroutine(coro).result(timeout=timeout)


def start_runtime(cfg: dict) -> AppRuntime:
    """Construct and start every async subsystem from `config.yaml` values.

    Engagement collection still uses `MockEngagementClient` - wiring a real
    Twitter/LinkedIn/Facebook analytics client is a one-line change here
    (`EngagementCollector(bus=bus, client=<real client>, ...)`) once one
    exists; `EngagementCollector` was designed against the `EngagementClient`
    Protocol precisely so this file is the only thing that would need to
    change.

    Raises `AutoPosterError` (not a raw `RuntimeError`) if the background
    loop fails to start, so `main.py`'s existing `except AutoPosterError`
    handling in `__main__` covers this failure mode without any changes
    there.
    """
    sd_cfg = cfg.get("semantic_dedup", {}) or {}
    fl_cfg = cfg.get("feedback_loop", {}) or {}

    dedup_config = SemanticDedupConfig(
        semantic_similarity_threshold=sd_cfg.get("similarity_threshold", 0.86),
        max_index_size=cfg.get("recent_topics_window", 30),
        enable_api_fallback=sd_cfg.get("api_fallback", True),
    )
    api_key = env("OPENAI_API_KEY") if sd_cfg.get("api_fallback", True) else None
    deduplicator = create_deduplicator(api_key=api_key, config=dedup_config)

    bus = EventBus()
    collector = EngagementCollector(
        bus=bus,
        client=MockEngagementClient(),
        config=EngagementCollectorConfig(
            poll_interval_seconds=fl_cfg.get("poll_interval_seconds", 300.0),
            max_snapshots_per_post=fl_cfg.get("max_snapshots_per_post", 5),
        ),
    )
    subscriber = FeedbackSubscriber(
        bus=bus,
        repository=build_memory_repository(),
        config=FeedbackSubscriberConfig(
            history_window=fl_cfg.get("history_window", 200),
            exemplars_per_label=fl_cfg.get("exemplars_per_label", 2),
        ),
    )

    manager = LifecycleManager()
    try:
        manager.start(bus, collector, subscriber)
    except Exception as exc:
        raise AutoPosterError(f"Failed to start the background event loop: {exc}") from exc

    return AppRuntime(
        manager=manager,
        bus=bus,
        deduplicator=deduplicator,
        collector=collector,
        subscriber=subscriber,
    )
