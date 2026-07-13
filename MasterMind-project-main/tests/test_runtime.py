"""Tests for the AppRuntime composition root (src/runtime.py).

`start_runtime()`'s real wiring uses `SemanticDeduplicator`'s default local
embedder (fastembed) and a FAISS index - optional heavy dependencies (see
requirements-semantic.txt) that may not be installed in every environment.
These tests patch `create_deduplicator` with a fake `SemanticDeduplicator`
built from the same plain-Python fakes used in test_semantic_dedup.py, so
this suite never requires faiss/fastembed to be installed. The dedup
cascade itself is already covered by test_semantic_dedup.py; what's under
test here is the *wiring* and the synchronous facade.
"""

from unittest.mock import patch

import numpy as np
import pytest

from src.event_bus import PostPublished
from src.exceptions import AutoPosterError
from src.feedback_loop import LifecycleManager
from src.runtime import AppRuntime, start_runtime
from src.semantic_dedup import SemanticDeduplicator


class _FakeEmbedder:
    def __init__(self, dimension=4):
        self._dimension = dimension

    @property
    def name(self):
        return "fake"

    @property
    def dimension(self):
        return self._dimension

    async def embed(self, texts):
        return np.zeros((len(texts), self._dimension), dtype=np.float32)


class _FakeIndex:
    def __init__(self):
        self._store = {}

    def add(self, item_id, vector):
        self._store[item_id] = vector

    def remove(self, item_id):
        self._store.pop(item_id, None)

    def search(self, query, top_k=1):
        return []

    def __len__(self):
        return len(self._store)


def _fake_deduplicator(**_kwargs):
    return SemanticDeduplicator(primary_embedder=_FakeEmbedder(), index=_FakeIndex())


def _base_cfg(**overrides):
    cfg = {
        "recent_topics_window": 10,
        "semantic_dedup": {"enabled": True, "similarity_threshold": 0.9, "api_fallback": False},
        "feedback_loop": {"poll_interval_seconds": 9999, "max_snapshots_per_post": 5},
    }
    cfg.update(overrides)
    return cfg


def test_start_runtime_wires_everything_and_shuts_down_cleanly():
    with patch("src.runtime.create_deduplicator", side_effect=_fake_deduplicator):
        runtime = start_runtime(_base_cfg())
    try:
        assert isinstance(runtime, AppRuntime)
        assert runtime.manager.is_running is True
    finally:
        runtime.shutdown()
    assert runtime.manager.is_running is False


def test_app_runtime_sync_facade_round_trips_through_the_shared_loop():
    with patch("src.runtime.create_deduplicator", side_effect=_fake_deduplicator):
        runtime = start_runtime(_base_cfg())
    try:
        runtime.hydrate_topics_sync(["old topic one", "old topic two"])

        verdict = runtime.check_topic_sync("a brand new topic")
        assert verdict.is_duplicate is False

        runtime.publish_post_published_sync(
            PostPublished(platform="twitter", post_id="p1", topic="AI", text="hi #ai")
        )
        runtime.manager.run_coroutine(runtime.bus.flush()).result(timeout=5.0)
        assert runtime.collector.tracked_count == 1

        exemplars = runtime.get_few_shot_exemplars_sync("twitter")
        assert exemplars == []  # no engagement recorded yet - not an error
    finally:
        runtime.shutdown()


def test_start_runtime_raises_autoposter_error_when_loop_fails_to_start():
    with (
        patch("src.runtime.create_deduplicator", side_effect=_fake_deduplicator),
        patch.object(LifecycleManager, "start", side_effect=RuntimeError("boom")),
        pytest.raises(AutoPosterError),
    ):
        start_runtime(_base_cfg())
