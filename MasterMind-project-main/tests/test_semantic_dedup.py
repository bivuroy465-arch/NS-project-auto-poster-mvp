"""Tests for the hybrid lexical + semantic dedup subsystem.

`SemanticDeduplicator` depends only on the `EmbeddingProvider`/`VectorIndex`
Protocols, so these tests never import faiss/fastembed/openai - plain Python
fakes are enough to exercise the full cascade (lexical short-circuit,
local-embedding hit, API fallback, fail-open) and the sliding-window
eviction logic. That is the entire point of designing the module with
Protocols and constructor injection.
"""

import asyncio

import numpy as np
import pytest
from pydantic import ValidationError

from src.semantic_dedup import (
    DedupMethod,
    EmbeddingProviderDegraded,
    SemanticDedupConfig,
    SemanticDeduplicator,
)


class _FakeEmbedder:
    """Deterministic embedder: returns pre-registered vectors, or zeros."""

    def __init__(self, dimension=4, name="fake", vectors=None, fail=False):
        self._dimension = dimension
        self._name = name
        self._vectors = vectors or {}
        self.fail = fail
        self.calls: list[list[str]] = []

    @property
    def name(self):
        return self._name

    @property
    def dimension(self):
        return self._dimension

    async def embed(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("embedder boom")
        return np.array(
            [self._vectors.get(t, [0.0] * self._dimension) for t in texts],
            dtype=np.float32,
        )


class _FakeIndex:
    """Brute-force in-memory cosine index; no FAISS required for tests."""

    def __init__(self, dimension=4):
        self._dimension = dimension
        self._store: dict[int, np.ndarray] = {}

    def add(self, item_id, vector):
        norm = np.linalg.norm(vector) or 1.0
        self._store[item_id] = vector / norm

    def remove(self, item_id):
        self._store.pop(item_id, None)

    def search(self, query, top_k=1):
        if not self._store:
            return []
        norm = np.linalg.norm(query) or 1.0
        q = query / norm
        scored = [(i, float(np.dot(q, v))) for i, v in self._store.items()]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def __len__(self):
        return len(self._store)


class _FakePublisher:
    def __init__(self):
        self.events = []

    async def publish(self, event):
        self.events.append(event)


def _dedup(**overrides):
    primary = overrides.pop("primary", _FakeEmbedder())
    fallback = overrides.pop("fallback", None)
    config = overrides.pop("config", None)
    publisher = overrides.pop("publisher", None)
    index = overrides.pop("index", _FakeIndex())
    return SemanticDeduplicator(
        primary_embedder=primary,
        index=index,
        fallback_embedder=fallback,
        config=config,
        event_publisher=publisher,
    )


def test_lexical_short_circuit_skips_embeddings():
    primary = _FakeEmbedder()
    dedup = _dedup(primary=primary)

    async def _run():
        await dedup.add("Best productivity tips for developers")
        return await dedup.check("best productivity tips for developers")

    verdict = asyncio.run(_run())
    assert verdict.is_duplicate is True
    assert verdict.method == DedupMethod.LEXICAL
    assert primary.calls == [["Best productivity tips for developers"]]  # only the add() call


def test_semantic_duplicate_detected_via_local_embedder():
    vectors = {
        "Why edge AI is taking over": [1.0, 0.0, 0.0, 0.0],
        "Edge AI is eating the world": [0.99, 0.01, 0.0, 0.0],
    }
    primary = _FakeEmbedder(vectors=vectors)
    dedup = _dedup(primary=primary, config=SemanticDedupConfig(semantic_similarity_threshold=0.9))

    async def _run():
        await dedup.add("Why edge AI is taking over")
        return await dedup.check("Edge AI is eating the world")

    verdict = asyncio.run(_run())
    assert verdict.is_duplicate is True
    assert verdict.method == DedupMethod.SEMANTIC_LOCAL
    assert verdict.matched_text == "Why edge AI is taking over"


def test_distinct_topics_are_not_duplicates_and_get_remembered():
    vectors = {
        "Quantum computing basics": [1.0, 0.0, 0.0, 0.0],
        "Best coffee brewing methods": [0.0, 1.0, 0.0, 0.0],
    }
    primary = _FakeEmbedder(vectors=vectors)
    dedup = _dedup(primary=primary)

    async def _run():
        await dedup.add("Quantum computing basics")
        return await dedup.check("Best coffee brewing methods")

    verdict = asyncio.run(_run())
    assert verdict.is_duplicate is False
    assert len(dedup) == 2  # the unique candidate was remembered


def test_falls_back_to_api_embedder_when_local_fails():
    primary = _FakeEmbedder(fail=True, name="local")
    fallback_vectors = {
        "Edge AI is eating the world": [0.99, 0.01, 0.0, 0.0],
        "Why edge AI is taking over": [1.0, 0.0, 0.0, 0.0],
    }
    fallback = _FakeEmbedder(vectors=fallback_vectors, name="api")
    dedup = _dedup(
        primary=primary,
        fallback=fallback,
        config=SemanticDedupConfig(semantic_similarity_threshold=0.9),
    )

    async def _run():
        await dedup.add("Why edge AI is taking over")
        return await dedup.check("Edge AI is eating the world")

    verdict = asyncio.run(_run())
    assert verdict.is_duplicate is True
    assert verdict.method == DedupMethod.SEMANTIC_API


def test_fails_open_when_no_embedder_is_available():
    primary = _FakeEmbedder(fail=True)
    dedup = _dedup(primary=primary, fallback=None)

    verdict = asyncio.run(dedup.check("Some brand new topic"))
    assert verdict.is_duplicate is False
    assert verdict.method == DedupMethod.DEGRADED


def test_fails_open_when_both_primary_and_fallback_fail():
    primary = _FakeEmbedder(fail=True, name="local")
    fallback = _FakeEmbedder(fail=True, name="api")
    dedup = _dedup(primary=primary, fallback=fallback)

    verdict = asyncio.run(dedup.check("Yet another topic"))
    assert verdict.is_duplicate is False
    assert verdict.method == DedupMethod.DEGRADED


def test_api_fallback_disabled_via_config_even_if_configured():
    primary = _FakeEmbedder(fail=True)
    fallback = _FakeEmbedder(name="api")
    config = SemanticDedupConfig(enable_api_fallback=False)
    dedup = _dedup(primary=primary, fallback=fallback, config=config)

    verdict = asyncio.run(dedup.check("Another topic"))
    assert verdict.is_duplicate is False
    assert verdict.method == DedupMethod.DEGRADED
    assert fallback.calls == []


def test_publishes_events_for_every_check():
    publisher = _FakePublisher()
    dedup = _dedup(publisher=publisher)

    asyncio.run(dedup.check("A topic nobody has seen"))
    assert len(publisher.events) == 1
    assert publisher.events[0].candidate == "A topic nobody has seen"


def test_publishes_degraded_event_when_primary_fails_even_if_fallback_succeeds():
    publisher = _FakePublisher()
    primary = _FakeEmbedder(fail=True)
    fallback = _FakeEmbedder()
    dedup = _dedup(primary=primary, fallback=fallback, publisher=publisher)

    asyncio.run(dedup.check("Some topic"))

    degraded = [e for e in publisher.events if isinstance(e, EmbeddingProviderDegraded)]
    assert len(degraded) == 1
    assert degraded[0].provider == primary.name


def test_hydrate_evicts_oldest_first_when_over_capacity():
    config = SemanticDedupConfig(max_index_size=2)
    primary = _FakeEmbedder(
        vectors={
            "oldest": [1.0, 0.0, 0.0, 0.0],
            "middle": [0.0, 1.0, 0.0, 0.0],
            "newest": [0.0, 0.0, 1.0, 0.0],
        }
    )
    dedup = _dedup(primary=primary, config=config)

    # sheets_logger.recent_topics() returns newest-first.
    asyncio.run(dedup.hydrate(["newest", "middle", "oldest"]))

    assert len(dedup) == 2
    assert set(dedup.remembered_topics) == {"newest", "middle"}


def test_dimension_mismatch_between_strategies_is_rejected():
    with pytest.raises(ValueError, match="dimension"):
        _dedup(primary=_FakeEmbedder(dimension=4), fallback=_FakeEmbedder(dimension=8))


def test_config_rejects_incoherent_thresholds():
    with pytest.raises(ValidationError):
        SemanticDedupConfig(semantic_similarity_threshold=0.9, lexical_short_circuit_threshold=0.5)
