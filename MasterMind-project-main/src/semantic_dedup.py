"""Hybrid lexical + semantic duplicate-topic guard.

Extends `dedup.py`'s cheap Jaccard check with an embeddings-based semantic
layer that catches paraphrases plain token overlap misses (e.g. "The rise of
edge AI" vs. "Why AI is moving to the edge"). Designed as a cost-ordered
cascade, cheapest tier first:

    1. Lexical short-circuit (reuses `dedup.similarity`)        -- microseconds
    2. Local embeddings (ONNX, offline) + in-memory FAISS index -- ~1-5 ms
    3. API embeddings (OpenAI) -- ONLY if the local model fails  -- ~100-300 ms
    4. Fail open (treat as unique; never blocks the pipeline)   -- if both fail

Note tier 3 is gated on *local-model unavailability*, not on lexical
ambiguity. Paraphrases (the whole point of this module) typically have LOW
lexical overlap but HIGH semantic similarity, so gating the semantic tier
behind "lexical was ambiguous" would skip embeddings for exactly the cases
we need them for. The API is purely a reliability fallback, not a second
opinion.

Every moving part - the embedding model, the vector index, the event sink -
is a `typing.Protocol` supplied via constructor injection (Strategy pattern
+ DI), so `SemanticDeduplicator` is fully unit-testable with plain-Python
fakes and never needs faiss/fastembed/network access in tests. See
`tests/test_semantic_dedup.py`.

This module deliberately does not know about engagement tracking, Sheets, or
Telegram/Discord alerts, and never performs blocking I/O (e.g. it does not
call `notifier.send_alert` directly, since that would block the event loop).
It only publishes plain domain events through an injected `EventPublisher`.
That is the seam the self-learning feedback loop - and, separately, a sync
alerting adapter - hook into.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import itertools
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from enum import StrEnum
from types import ModuleType
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import dedup as _dedup
from .event_bus import DomainEvent
from .exceptions import ProviderError
from .logging_setup import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only, no hard runtime import
    from fastembed import TextEmbedding

log = get_logger("semantic_dedup")

__all__ = [
    "DedupMethod",
    "DomainEvent",
    "TopicDedupEvaluated",
    "EmbeddingProviderDegraded",
    "SemanticDedupConfig",
    "DedupVerdict",
    "EmbeddingProvider",
    "VectorIndex",
    "EventPublisher",
    "FastEmbedLocalProvider",
    "OpenAIEmbeddingProvider",
    "FaissCosineIndex",
    "SemanticDeduplicator",
    "create_local_embedder",
    "create_api_fallback_embedder",
    "create_deduplicator",
    "is_duplicate_sync",
]

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_API_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSION = 384


# --------------------------------------------------------------------------
# Enums & domain events
# --------------------------------------------------------------------------
class DedupMethod(StrEnum):
    """Which tier of the cascade produced a `DedupVerdict`."""

    LEXICAL = "lexical"
    SEMANTIC_LOCAL = "semantic-local"
    SEMANTIC_API = "semantic-api"
    DEGRADED = "degraded-fail-open"
    SKIPPED_EMPTY = "skipped-empty"


class DedupVerdict(BaseModel):
    """Rich, structured result of a duplicate check.

    Deliberately not a bare bool: callers (and observability subscribers)
    get the deciding tier, the match, the score, and the latency for free.
    """

    model_config = ConfigDict(frozen=True)

    is_duplicate: bool
    method: DedupMethod
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    matched_text: str | None = None
    corpus_size: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)


class TopicDedupEvaluated(DomainEvent):
    """Published after every `SemanticDeduplicator.check()` call."""

    candidate: str
    verdict: DedupVerdict


class EmbeddingProviderDegraded(DomainEvent):
    """Published whenever the primary embedder fails, win or lose on fallback."""

    provider: str
    error: str


class SemanticDedupConfig(BaseModel):
    """Immutable tuning knobs for the cascade."""

    model_config = ConfigDict(frozen=True)

    semantic_similarity_threshold: float = Field(
        default=0.86,
        ge=0.0,
        le=1.0,
        description="Cosine similarity at/above which two topics are treated as duplicates.",
    )
    lexical_short_circuit_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Jaccard token overlap at/above which we skip embeddings entirely.",
    )
    max_index_size: int = Field(
        default=100,
        gt=0,
        description="Sliding-window size of the in-memory index (mirrors recent_topics_window).",
    )
    enable_api_fallback: bool = Field(
        default=True,
        description="Whether the API embedder may be used when the local model is unavailable.",
    )

    @model_validator(mode="after")
    def _thresholds_are_coherent(self) -> SemanticDedupConfig:
        if self.lexical_short_circuit_threshold < self.semantic_similarity_threshold:
            raise ValueError(
                "lexical_short_circuit_threshold must be >= semantic_similarity_threshold: "
                "otherwise the 'cheap' tier would be stricter than the semantic tier it is "
                "meant to short-circuit, which defeats its purpose."
            )
        return self


# --------------------------------------------------------------------------
# Strategy interfaces (Protocol = structural typing, no inheritance needed)
# --------------------------------------------------------------------------
@runtime_checkable
class EmbeddingProvider(Protocol):
    """Strategy interface for turning text into vectors.

    Any object with this shape works. Swap implementations purely via
    constructor injection into `SemanticDeduplicator` - that is the Strategy
    pattern applied to embedding models.
    """

    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float32]: ...


@runtime_checkable
class VectorIndex(Protocol):
    """Strategy interface for the in-memory nearest-neighbour index."""

    def add(self, item_id: int, vector: npt.NDArray[np.float32]) -> None: ...
    def remove(self, item_id: int) -> None: ...
    def search(self, query: npt.NDArray[np.float32], top_k: int = 1) -> list[tuple[int, float]]: ...
    def __len__(self) -> int: ...


@runtime_checkable
class EventPublisher(Protocol):
    """Strategy/seam for the event-driven feedback loop to observe dedup
    activity without this module knowing anything about engagement
    tracking, Sheets, or notifications.
    """

    async def publish(self, event: DomainEvent) -> None: ...


class _NullEventPublisher:
    """Default no-op publisher so `event_publisher` is truly optional."""

    async def publish(self, event: DomainEvent) -> None:
        return None


def _lazy_import(module_name: str, *, extra_hint: str) -> ModuleType:
    """Import an optional heavy dependency with an actionable error message.

    faiss/fastembed are not core dependencies of the poster (they pull in
    real binary wheels), so importing them eagerly at module load time would
    force every user to install them just to import `semantic_dedup`. This
    keeps them opt-in and fails with a clear instruction instead of a bare
    `ModuleNotFoundError` three frames deep.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ProviderError(
            f"'{module_name}' is required for this feature but is not installed. "
            f"Install it with: pip install -r requirements-semantic.txt  (needs: {extra_hint})"
        ) from exc


# --------------------------------------------------------------------------
# Embedding strategies
# --------------------------------------------------------------------------
class FastEmbedLocalProvider:
    """Local, offline embedding strategy backed by ONNX Runtime (fastembed).

    Deliberately not sentence-transformers: fastembed ships ONNX models and
    has no torch dependency, which is the difference between a ~50-100 MB
    install and a 1 GB+ one - it matters for a small poster meant to run in
    a slim CI container. This is the "fast, in-memory local vector search"
    default: no network call, no per-request cost, sub-second even on a
    cold CI runner after the model is cached.

    CPU-bound ONNX inference is offloaded to a worker thread via
    `asyncio.to_thread` so it never blocks the event loop; ONNX Runtime
    releases the GIL during the actual matrix math, so this buys real
    concurrency, not just cooperative scheduling.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_LOCAL_MODEL,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._model: TextEmbedding | None = None
        # A real OS thread.Lock, not asyncio.Lock: `_embed_sync` runs inside
        # a thread-pool worker (via `to_thread`), not on the event-loop
        # thread, so only a threading primitive actually guards it.
        self._load_lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"fastembed:{self._model_name}"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        return await asyncio.to_thread(self._embed_sync, list(texts))

    def _embed_sync(self, texts: list[str]) -> npt.NDArray[np.float32]:
        model = self._load_model()
        return np.asarray(list(model.embed(texts)), dtype=np.float32)

    def _load_model(self) -> TextEmbedding:
        if self._model is None:
            with self._load_lock:  # double-checked locking around the lazy singleton
                if self._model is None:
                    fastembed = _lazy_import("fastembed", extra_hint="fastembed")
                    self._model = fastembed.TextEmbedding(model_name=self._model_name)
        return self._model


class OpenAIEmbeddingProvider:
    """API-backed embedding strategy - the reliability fallback, not a peer.

    Uses the async OpenAI client for genuine non-blocking network I/O.
    Requests the same `dimension` as the local model via the native
    `dimensions=` truncation supported by the `text-embedding-3-*` family,
    so both strategies populate one shared FAISS index with directly
    comparable vectors.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_API_MODEL,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        timeout: float = 10.0,
    ) -> None:
        from openai import AsyncOpenAI  # already a hard project dependency

        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self._model = model
        self._dimension = dimension

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        try:
            resp = await self._client.embeddings.create(
                model=self._model,
                input=list(texts),
                dimensions=self._dimension,
            )
        except Exception as exc:
            raise ProviderError(f"OpenAI embedding request failed: {exc}") from exc
        return np.array([row.embedding for row in resp.data], dtype=np.float32)


# --------------------------------------------------------------------------
# Vector index strategy
# --------------------------------------------------------------------------
class FaissCosineIndex:
    """Exact cosine-similarity index backed by FAISS `IndexFlatIP`.

    Deliberately exact, not approximate (no IVF/HNSW): the corpus this
    indexes is a sliding window of recent topics (dozens to a few hundred
    entries, per `SemanticDedupConfig.max_index_size`), where brute-force
    search is sub-millisecond and full recall matters far more than
    sub-linear scaling. If the corpus ever needs to grow past ~50k-100k
    vectors, swap in an approximate `VectorIndex` implementation - this
    class is just another Strategy behind the same Protocol, so nothing
    else in this module would need to change.

    Vectors are always re-normalised to unit length on insert and query,
    regardless of whether the embedder already normalises them (OpenAI's
    `dimensions`-truncated vectors, for instance, are not guaranteed to be
    unit-length) - so inner product here is always equivalent to cosine
    similarity.

    IDs are monotonically increasing and never reused (see
    `SemanticDeduplicator._next_id`), so a stale ID can never silently
    collide with a fresh insert after an eviction.
    """

    def __init__(self, dimension: int) -> None:
        faiss = _lazy_import("faiss", extra_hint="faiss-cpu")
        self._faiss = faiss
        self._dimension = dimension
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
        # Belt-and-suspenders: current call sites only ever touch this from
        # the event-loop thread, but guarding it means this class stays
        # safe to reuse from a multi-threaded context without the caller
        # needing to know our internals.
        self._lock = threading.Lock()

    def add(self, item_id: int, vector: npt.NDArray[np.float32]) -> None:
        vec = self._prepare(vector)
        with self._lock:
            self._index.add_with_ids(vec, np.array([item_id], dtype=np.int64))

    def remove(self, item_id: int) -> None:
        with self._lock:
            self._index.remove_ids(np.array([item_id], dtype=np.int64))

    def search(self, query: npt.NDArray[np.float32], top_k: int = 1) -> list[tuple[int, float]]:
        vec = self._prepare(query)
        with self._lock:
            if self._index.ntotal == 0:
                return []
            scores, ids = self._index.search(vec, min(top_k, self._index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i != -1]

    def _prepare(self, vector: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        vec = np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)
        self._faiss.normalize_L2(vec)
        return vec

    def __len__(self) -> int:
        return int(self._index.ntotal)


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
@dataclasses.dataclass(slots=True)
class _RuntimeStats:
    total_checks: int = 0
    lexical_short_circuits: int = 0
    semantic_local_hits: int = 0
    semantic_api_fallbacks: int = 0
    degraded_fail_opens: int = 0


class _LexicalHit(NamedTuple):
    text: str
    score: float


def _lexical_short_circuit(
    candidate: str, corpus: Sequence[str], threshold: float
) -> _LexicalHit | None:
    """Cheap pre-filter: catches exact/near-exact repeats before touching embeddings."""
    best: _LexicalHit | None = None
    for text in corpus:
        score = _dedup.similarity(candidate, text)
        if score >= threshold and (best is None or score > best.score):
            best = _LexicalHit(text=text, score=score)
    return best


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
class SemanticDeduplicator:
    """Hybrid lexical + semantic duplicate-topic guard.

    See the module docstring for the full cascade. Every dependency here is
    injected through the constructor and expressed as a `Protocol`, so this
    class is fully unit-testable with lightweight fakes - no FAISS,
    fastembed, or network access required in tests.
    """

    def __init__(
        self,
        *,
        primary_embedder: EmbeddingProvider,
        index: VectorIndex,
        fallback_embedder: EmbeddingProvider | None = None,
        config: SemanticDedupConfig | None = None,
        event_publisher: EventPublisher | None = None,
    ) -> None:
        if (
            fallback_embedder is not None
            and fallback_embedder.dimension != primary_embedder.dimension
        ):
            raise ValueError(
                "primary_embedder and fallback_embedder must share a vector dimension to "
                f"populate one shared index (got {primary_embedder.dimension} vs "
                f"{fallback_embedder.dimension}). Configure the fallback's `dimensions=` "
                "to match the local model."
            )
        self._primary = primary_embedder
        self._fallback = fallback_embedder
        self._index = index
        self._config = config or SemanticDedupConfig()
        self._events = event_publisher or _NullEventPublisher()
        self._corpus: OrderedDict[int, str] = OrderedDict()  # id -> text, FIFO recency order
        self._text_to_id: dict[str, int] = {}  # O(1) membership / idempotent inserts
        self._next_id = itertools.count(1)  # monotonic: freed ids are never reused
        # Guards `_corpus`/`_text_to_id`/`_next_id` across `await` points -
        # unlike the index's threading.Lock, mutations here interleave with
        # real awaits (embedding calls), so this must be an asyncio.Lock.
        self._lock = asyncio.Lock()
        self._stats = _RuntimeStats()

    @property
    def stats(self) -> _RuntimeStats:
        """Read-only snapshot; mutating the result does not affect internal counters."""
        return dataclasses.replace(self._stats)

    @property
    def remembered_topics(self) -> tuple[str, ...]:
        """Snapshot of the in-memory corpus, oldest first."""
        return tuple(self._corpus.values())

    def __len__(self) -> int:
        return len(self._corpus)

    async def check(self, candidate: str) -> DedupVerdict:
        """Return a `DedupVerdict` for `candidate` against the recent-topics window.

        As a side effect, unique candidates are remembered for future
        checks; rejected duplicates are not (a rejected topic is never
        actually published, so it should not pollute the corpus).
        """
        start = time.perf_counter()
        self._stats.total_checks += 1
        text = candidate.strip()
        if not text:
            return DedupVerdict(
                is_duplicate=False,
                method=DedupMethod.SKIPPED_EMPTY,
                corpus_size=len(self._corpus),
                latency_ms=_elapsed_ms(start),
            )

        async with self._lock:
            snapshot = list(self._corpus.values())

        lexical_hit = _lexical_short_circuit(
            text, snapshot, self._config.lexical_short_circuit_threshold
        )
        if lexical_hit is not None:
            self._stats.lexical_short_circuits += 1
            verdict = DedupVerdict(
                is_duplicate=True,
                method=DedupMethod.LEXICAL,
                score=lexical_hit.score,
                matched_text=lexical_hit.text,
                corpus_size=len(snapshot),
                latency_ms=_elapsed_ms(start),
            )
            await self._publish(text, verdict)
            return verdict

        try:
            vector, method = await self._embed_one_with_fallback(text)
        except Exception as exc:
            # Fail open, matching `image_generator.generate_image`'s and
            # `sheets_logger.recent_topics`'s "best-effort, never break the
            # run" contract elsewhere in this codebase.
            self._stats.degraded_fail_opens += 1
            log.error("Semantic dedup degraded to fail-open (no embedder available): %s", exc)
            verdict = DedupVerdict(
                is_duplicate=False,
                method=DedupMethod.DEGRADED,
                corpus_size=len(snapshot),
                latency_ms=_elapsed_ms(start),
            )
            await self._publish(text, verdict)
            return verdict

        matches = self._index.search(vector, top_k=1)
        verdict = self._build_verdict(matches[0] if matches else None, method, snapshot, start)
        if not verdict.is_duplicate:
            async with self._lock:
                self._insert_locked(text, vector)
        await self._publish(text, verdict)
        return verdict

    async def add(self, text: str) -> None:
        """Unconditionally embed and remember `text`, skipping the duplicate check.

        Useful for seeding the index from a source other than `check()`
        (e.g. a manually curated topic list).
        """
        text = text.strip()
        if not text:
            return
        vector, _method = await self._embed_one_with_fallback(text)
        async with self._lock:
            self._insert_locked(text, vector)

    async def hydrate(self, recent_topics: Sequence[str]) -> None:
        """Bulk-load existing topics into the index once at startup.

        `recent_topics` is expected newest-first (matching
        `sheets_logger.recent_topics()`); we insert oldest-first so the
        sliding-window eviction in `_evict_if_needed_locked` drops the
        least-recent topics first, rather than an arbitrary artifact of
        insertion order.
        """
        chronological = list(reversed(recent_topics))
        async with self._lock:
            unseen = [t for t in chronological if t not in self._text_to_id]
        if not unseen:
            return
        vectors, _method = await self._embed_with_fallback(unseen)
        async with self._lock:
            for text, vector in zip(unseen, vectors, strict=True):
                self._insert_locked(text, vector)

    async def _embed_one_with_fallback(
        self, text: str
    ) -> tuple[npt.NDArray[np.float32], DedupMethod]:
        vectors, method = await self._embed_with_fallback([text])
        return vectors[0], method

    async def _embed_with_fallback(
        self, texts: Sequence[str]
    ) -> tuple[npt.NDArray[np.float32], DedupMethod]:
        try:
            vectors = await self._primary.embed(texts)
        except Exception as exc:
            log.warning("Primary embedder %r failed (%s).", self._primary.name, exc)
            await self._publish_degraded(exc)
            if self._fallback is None or not self._config.enable_api_fallback:
                raise ProviderError(
                    f"Local embedder {self._primary.name!r} failed and no API fallback is "
                    f"configured/enabled: {exc}"
                ) from exc
            vectors = await self._fallback.embed(texts)
            self._stats.semantic_api_fallbacks += 1
            return vectors, DedupMethod.SEMANTIC_API
        else:
            self._stats.semantic_local_hits += 1
            return vectors, DedupMethod.SEMANTIC_LOCAL

    def _build_verdict(
        self,
        match: tuple[int, float] | None,
        method: DedupMethod,
        snapshot: list[str],
        start: float,
    ) -> DedupVerdict:
        if match is None:
            return DedupVerdict(
                is_duplicate=False,
                method=method,
                corpus_size=len(snapshot),
                latency_ms=_elapsed_ms(start),
            )
        item_id, score = match
        is_dup = score >= self._config.semantic_similarity_threshold
        return DedupVerdict(
            is_duplicate=is_dup,
            method=method,
            score=score,
            matched_text=self._corpus.get(item_id) if is_dup else None,
            corpus_size=len(snapshot),
            latency_ms=_elapsed_ms(start),
        )

    def _insert_locked(self, text: str, vector: npt.NDArray[np.float32]) -> None:
        """Caller must hold `self._lock`."""
        if text in self._text_to_id:
            return
        item_id = next(self._next_id)
        self._corpus[item_id] = text
        self._text_to_id[text] = item_id
        self._index.add(item_id, vector)
        self._evict_if_needed_locked()

    def _evict_if_needed_locked(self) -> None:
        """Caller must hold `self._lock`."""
        while len(self._corpus) > self._config.max_index_size:
            oldest_id, oldest_text = self._corpus.popitem(last=False)  # FIFO: oldest first
            del self._text_to_id[oldest_text]
            self._index.remove(oldest_id)

    async def _publish(self, candidate: str, verdict: DedupVerdict) -> None:
        try:
            await self._events.publish(TopicDedupEvaluated(candidate=candidate, verdict=verdict))
        except Exception:
            log.debug("Event publish failed; continuing.", exc_info=True)

    async def _publish_degraded(self, exc: Exception) -> None:
        try:
            await self._events.publish(
                EmbeddingProviderDegraded(provider=self._primary.name, error=str(exc))
            )
        except Exception:
            log.debug("Event publish failed; continuing.", exc_info=True)


# --------------------------------------------------------------------------
# Factories (mirrors the `providers/*/factory.py` convention elsewhere in
# this codebase) and a transitional sync bridge
# --------------------------------------------------------------------------
def create_local_embedder(
    *,
    model_name: str = DEFAULT_LOCAL_MODEL,
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
) -> EmbeddingProvider:
    """Strategy factory: local, offline, no per-call cost."""
    return FastEmbedLocalProvider(model_name=model_name, dimension=dimension)


def create_api_fallback_embedder(
    *,
    api_key: str,
    model: str = DEFAULT_API_MODEL,
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    timeout: float = 10.0,
) -> EmbeddingProvider:
    """Strategy factory: the reliability fallback, only used if the local model fails."""
    return OpenAIEmbeddingProvider(
        api_key=api_key, model=model, dimension=dimension, timeout=timeout
    )


def create_deduplicator(
    *,
    api_key: str | None = None,
    config: SemanticDedupConfig | None = None,
    event_publisher: EventPublisher | None = None,
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
) -> SemanticDeduplicator:
    """Convenience wiring for the recommended default strategy chain.

    Equivalent in spirit to `providers.image.factory.get_image_chain`:
    builds a sensible default pipeline, while `SemanticDeduplicator` itself
    remains fully swappable by constructing it directly with different
    strategies.
    """
    primary = create_local_embedder(dimension=dimension)
    fallback = (
        create_api_fallback_embedder(api_key=api_key, dimension=dimension) if api_key else None
    )
    index = FaissCosineIndex(dimension=dimension)
    return SemanticDeduplicator(
        primary_embedder=primary,
        index=index,
        fallback_embedder=fallback,
        config=config,
        event_publisher=event_publisher,
    )


def is_duplicate_sync(deduplicator: SemanticDeduplicator, candidate: str) -> DedupVerdict:
    """Transitional bridge for today's synchronous call sites (`main.py`,
    `topic_generator.py`).

    Prefer `await deduplicator.check(...)` from any async caller - this
    exists purely so the async-native core above can be adopted
    incrementally without a big-bang rewrite of the orchestrator. Cannot be
    called from within a running event loop (raises `RuntimeError`), since
    it starts its own via `asyncio.run`.
    """
    return asyncio.run(deduplicator.check(candidate))
