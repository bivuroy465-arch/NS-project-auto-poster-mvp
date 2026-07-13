"""Normalisation and similarity helpers for the topic duplicate guard."""
import re

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalise(text: str) -> str:
    """Lowercase and collapse to significant word tokens."""
    return " ".join(_WORD_RE.findall(text.lower()))


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def similarity(a: str, b: str) -> float:
    """Jaccard similarity over word tokens (0.0 - 1.0)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union


def is_duplicate(candidate: str, previous: list[str], threshold: float = 0.6) -> bool:
    """True if candidate exactly matches or is highly similar to any previous topic."""
    norm = normalise(candidate)
    if not norm:
        return False
    for prev in previous:
        if normalise(prev) == norm:
            return True
        if similarity(candidate, prev) >= threshold:
            return True
    return False
