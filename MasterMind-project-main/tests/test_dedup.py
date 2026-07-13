"""Unit tests for the duplicate-guard helpers."""
from src.dedup import normalise, similarity, is_duplicate


def test_normalise_strips_punctuation_and_case():
    assert normalise("  AI & Productivity!! ") == "ai productivity"


def test_similarity_identical_is_one():
    assert similarity("machine learning tips", "machine learning tips") == 1.0


def test_similarity_disjoint_is_zero():
    assert similarity("cats", "databases") == 0.0


def test_is_duplicate_exact_match():
    assert is_duplicate("AI productivity", ["ai productivity"]) is True


def test_is_duplicate_high_overlap():
    assert is_duplicate(
        "top productivity tips for developers",
        ["productivity tips for developers today"],
        threshold=0.6,
    ) is True


def test_is_not_duplicate_when_distinct():
    assert is_duplicate(
        "quantum computing basics",
        ["best coffee recipes", "travel in japan"],
    ) is False


def test_is_duplicate_empty_candidate_is_false():
    assert is_duplicate("", ["anything"]) is False
