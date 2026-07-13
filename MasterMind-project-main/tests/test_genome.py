"""Tests for the Genome data model and its repository Protocol."""

import asyncio

import pytest
from pydantic import ValidationError

from src.genome import Genome, InMemoryGenomeRepository


def test_create_computes_a_content_address():
    genome = Genome.create(hook_style="question", temperature=0.9)
    assert genome.genome_id
    assert len(genome.genome_id) == 64  # sha256 hex digest


def test_identical_strategy_fields_produce_the_identical_id():
    a = Genome.create(hook_style="story", temperature=0.5, hashtag_density=0.2)
    b = Genome.create(hook_style="story", temperature=0.5, hashtag_density=0.2)
    assert a.genome_id == b.genome_id


def test_identical_strategy_but_different_lineage_still_collapses_to_the_same_id():
    """Identity is about the strategy, not provenance: two lineages that
    converge on the same strategy should be treated as the same genome.
    """
    a = Genome.create(hook_style="story", temperature=0.5, parent_ids=("p1",), generation=3)
    b = Genome.create(hook_style="story", temperature=0.5, parent_ids=("p2", "p3"), generation=7)
    assert a.genome_id == b.genome_id


@pytest.mark.parametrize(
    "field,value",
    [
        ("hook_style", "controversy"),
        ("structural_template", "cta_first"),
        ("temperature", 1.1),
        ("hashtag_density", 0.9),
        ("exemplar_weighting", "recency"),
    ],
)
def test_any_strategy_field_change_produces_a_different_id(field, value):
    baseline = Genome.create()
    changed = Genome.create(**{field: value})
    assert baseline.genome_id != changed.genome_id


def test_genome_is_frozen():
    genome = Genome.create()
    with pytest.raises(ValidationError):
        genome.temperature = 1.2  # type: ignore[misc]


def test_temperature_out_of_range_is_rejected():
    with pytest.raises(ValidationError):
        Genome.create(temperature=5.0)


def test_hashtag_density_out_of_range_is_rejected():
    with pytest.raises(ValidationError):
        Genome.create(hashtag_density=-0.1)


def test_default_genome_is_constructible_with_no_arguments():
    genome = Genome.create()
    assert genome.generation == 0
    assert genome.parent_ids == ()


def test_repository_round_trips_a_genome():
    async def _run():
        repo = InMemoryGenomeRepository()
        genome = Genome.create(hook_style="statistic")
        await repo.save(genome)
        return await repo.get(genome.genome_id)

    fetched = asyncio.run(_run())
    assert fetched is not None
    assert fetched.hook_style == "statistic"


def test_repository_get_returns_none_for_unknown_id():
    result = asyncio.run(InMemoryGenomeRepository().get("does-not-exist"))
    assert result is None


def test_repository_all_returns_every_saved_genome():
    async def _run():
        repo = InMemoryGenomeRepository()
        await repo.save(Genome.create(hook_style="question"))
        await repo.save(Genome.create(hook_style="story"))
        return await repo.all()

    genomes = asyncio.run(_run())
    assert {g.hook_style for g in genomes} == {"question", "story"}


def test_repository_save_is_idempotent_for_the_same_strategy():
    async def _run():
        repo = InMemoryGenomeRepository()
        await repo.save(Genome.create(hook_style="question"))
        await repo.save(Genome.create(hook_style="question"))  # identical id
        return await repo.all()

    genomes = asyncio.run(_run())
    assert len(genomes) == 1


def test_repository_delete_removes_a_genome():
    async def _run():
        repo = InMemoryGenomeRepository()
        genome = Genome.create()
        await repo.save(genome)
        await repo.delete(genome.genome_id)
        return await repo.get(genome.genome_id)

    assert asyncio.run(_run()) is None


def test_repository_delete_of_unknown_id_does_not_raise():
    asyncio.run(InMemoryGenomeRepository().delete("does-not-exist"))
