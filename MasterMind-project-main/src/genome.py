"""Content strategy genomes for the Ouroboros policy engine.

Phase 1 of docs/whitepapers/ouroboros-policy-engine.md: pure data
structures only. No event bus, no LLM calls, no wiring into main.py.

A `Genome` is a small, structured, *content-addressed* description of a
content-generation strategy - one "arm" the bandit in `bayesian_bandit.py`
chooses between. Its id is a hash of its own strategy fields, exactly like
a Git blob: two genomes with identical strategy fields are, by
construction, the same genome, regardless of how or when each was
created. That is deliberate, not incidental - if two independent mutation
events ever converge on the same strategy, they should collapse into one
population entry with pooled statistics, not silently fragment the
bandit's evidence across two redundant, near-identical arms.

Deliberately genome-agnostic math lives elsewhere: nothing in
`bayesian_bandit.py` ever inspects a `Genome`'s fields. The bandit only
ever sees an opaque `genome_id: str` and a numeric reward, so the specific
vocabulary chosen for `hook_style`/`structural_template`/etc. below can
change at any time without touching a single line of the bandit math or
its test suite.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Genome", "GenomeRepository", "InMemoryGenomeRepository"]

# The fields that define a strategy's *identity*. Lineage metadata
# (parent_ids, generation) is provenance, not identity - it is
# deliberately excluded from the content hash (see Genome.create).
_STRATEGY_FIELDS = (
    "hook_style",
    "structural_template",
    "temperature",
    "hashtag_density",
    "exemplar_weighting",
)


class Genome(BaseModel):
    """A single content-generation strategy - one arm of the bandit."""

    model_config = ConfigDict(frozen=True)

    genome_id: str
    parent_ids: tuple[str, ...] = ()
    generation: int = Field(default=0, ge=0)

    hook_style: Literal["question", "statistic", "bold_claim", "story", "controversy"] = (
        "bold_claim"
    )
    structural_template: str = "hook_insight_cta"
    temperature: float = Field(default=0.7, ge=0.0, le=1.5)
    hashtag_density: float = Field(default=0.5, ge=0.0, le=1.0)
    exemplar_weighting: Literal["recency", "performance", "uniform"] = "performance"

    @staticmethod
    def _hash_strategy(strategy: dict) -> str:
        """Deterministic content hash of the strategy-defining fields only."""
        encoded = json.dumps(strategy, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @classmethod
    def create(
        cls,
        *,
        hook_style: Literal[
            "question", "statistic", "bold_claim", "story", "controversy"
        ] = "bold_claim",
        structural_template: str = "hook_insight_cta",
        temperature: float = 0.7,
        hashtag_density: float = 0.5,
        exemplar_weighting: Literal["recency", "performance", "uniform"] = "performance",
        parent_ids: tuple[str, ...] = (),
        generation: int = 0,
    ) -> Genome:
        """The supported way to construct a Genome.

        Computes the content-addressed `genome_id` from the strategy
        fields automatically, so identical strategies always collapse to
        the identical id regardless of lineage. Deserializing a genome
        already known to a repository should use the normal
        `Genome(...)` constructor with an explicit `genome_id` instead of
        recomputing it.
        """
        strategy = {
            "hook_style": hook_style,
            "structural_template": structural_template,
            "temperature": temperature,
            "hashtag_density": hashtag_density,
            "exemplar_weighting": exemplar_weighting,
        }
        genome_id = cls._hash_strategy(strategy)
        return cls(
            genome_id=genome_id,
            parent_ids=tuple(parent_ids),
            generation=generation,
            **strategy,
        )


@runtime_checkable
class GenomeRepository(Protocol):
    """Durable-state Strategy for the genome population.

    Mirrors `feedback_loop.MemoryRepository`'s shape and philosophy
    exactly: async so a future Sheets/DB-backed implementation can offload
    blocking I/O via `asyncio.to_thread` without changing this interface,
    the same pattern `feedback_loop.SheetsMemoryRepository` already uses.
    """

    async def save(self, genome: Genome) -> None: ...
    async def get(self, genome_id: str) -> Genome | None: ...
    async def all(self) -> list[Genome]: ...
    async def delete(self, genome_id: str) -> None: ...


class InMemoryGenomeRepository:
    """In-memory GenomeRepository: the Phase 1 default, and the fake used
    throughout this subsystem's own test suite. A durable, Sheets-backed
    implementation is future work, mirroring
    `feedback_loop.SheetsMemoryRepository`.
    """

    def __init__(self) -> None:
        self._genomes: dict[str, Genome] = {}
        self._lock = asyncio.Lock()

    async def save(self, genome: Genome) -> None:
        async with self._lock:
            self._genomes[genome.genome_id] = genome

    async def get(self, genome_id: str) -> Genome | None:
        async with self._lock:
            return self._genomes.get(genome_id)

    async def all(self) -> list[Genome]:
        async with self._lock:
            return list(self._genomes.values())

    async def delete(self, genome_id: str) -> None:
        async with self._lock:
            self._genomes.pop(genome_id, None)
