"""Asks the AI to pick a fresh, relevant topic, avoiding recent duplicates."""
from .providers.text.base import TextProvider
from .dedup import is_duplicate
from .logging_setup import get_logger

log = get_logger("topic")


def _build_prompt(themes: list[str] | None, avoid: list[str] | None) -> str:
    parts = ["Pick ONE specific, engaging social media topic for a post today."]
    if themes:
        parts.append(f"Prefer topics related to: {', '.join(themes)}.")
    if avoid:
        recent = "; ".join(avoid[:30])
        parts.append(
            "Do NOT repeat or closely resemble any of these recent topics: "
            f"{recent}."
        )
    parts.append("Reply with only the topic as a short phrase, no explanation.")
    return " ".join(parts)


def generate_topic(
    provider: TextProvider,
    themes: list[str] | None = None,
    recent: list[str] | None = None,
    max_attempts: int = 3,
) -> str:
    """Generate a topic that does not duplicate `recent` topics.

    Retries up to `max_attempts`; returns the last candidate even if a unique
    one could not be found, so the run still proceeds.
    """
    recent = recent or []
    prompt = _build_prompt(themes, recent)
    candidate = ""
    for attempt in range(1, max_attempts + 1):
        candidate = (provider.generate(prompt, max_tokens=60) or "").strip()
        if not is_duplicate(candidate, recent):
            return candidate
        log.warning(
            "Topic attempt %d duplicated a recent topic: %r", attempt, candidate
        )
    log.warning("Could not find a unique topic after %d attempts; using last.", max_attempts)
    return candidate
