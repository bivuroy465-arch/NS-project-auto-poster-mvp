"""Content Quality Gate: a platform-aware rule layer plus an AI editor.

Fully decoupled from publishers and the orchestrator: it only depends on a
text provider (for the AI editor) and plain config values, so it can be unit
tested in isolation.

Returns a QualityResult(passed, reason). Callers decide what to do with it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .logging_setup import get_logger
from .providers.text.base import TextProvider

log = get_logger("quality_gate")

# Phrases that betray an LLM or unfinished content.
_DEFAULT_AI_ISMS = [
    "as an ai",
    "as a language model",
    "as an ai language model",
    "here is your post",
    "here's your post",
    "here is a post",
    "in conclusion",
    "overall,",
    "i cannot",
    "i'm sorry",
]
# Placeholder patterns like [insert link], {topic}, <name>.
_PLACEHOLDER_RE = re.compile(r"[\[{<][^\]}>]*\b(insert|your|link|name|topic|todo)\b[^\]}>]*[\]}>]", re.I)
# Platforms that should contain at least one hashtag.
_REQUIRE_HASHTAG = {"twitter"}


@dataclass
class QualityResult:
    passed: bool
    reason: str = ""


def check_rules(
    text: str,
    platform: str,
    limit: int,
    banned_words: list[str] | None = None,
) -> QualityResult:
    """Fast, deterministic, platform-aware checks. No network calls."""
    if not text or not text.strip():
        return QualityResult(False, "empty content")
    if len(text) > limit:
        return QualityResult(False, f"exceeds {platform} limit ({len(text)}>{limit})")
    lowered = text.lower()
    for ai in _DEFAULT_AI_ISMS:
        if ai in lowered:
            return QualityResult(False, f"contains AI-ism: {ai!r}")
    if _PLACEHOLDER_RE.search(text):
        return QualityResult(False, "contains a placeholder")
    for word in banned_words or []:
        if word and word.lower() in lowered:
            return QualityResult(False, f"contains banned word: {word!r}")
    if platform in _REQUIRE_HASHTAG and "#" not in text:
        return QualityResult(False, "missing required hashtag")
    return QualityResult(True)


_EDITOR_SYSTEM = (
    "You are a ruthless social media editor. You approve a post ONLY if it is "
    "publish-ready, natural, human-sounding and on-topic. You MUST reject it if "
    "it contains any AI-isms (e.g. 'Here is your post', 'As an AI language model', "
    "'In conclusion', 'Overall'), a robotic or generic tone, hallucinated or "
    "unverifiable claims, or placeholders (e.g. '[insert link]'). "
    "Reply with EXACTLY 'PASS' if acceptable, otherwise 'FAIL: <short reason>'."
)


def ai_review(provider: TextProvider, text: str, platform: str, topic: str) -> QualityResult:
    """Ask the editor provider to judge the post. Fails closed on bad output."""
    prompt = (
        f"Platform: {platform}\nTopic: {topic}\n\nPost:\n{text}\n\n"
        "Judge this post now."
    )
    verdict = (provider.generate(prompt, system=_EDITOR_SYSTEM, max_tokens=60) or "").strip()
    if verdict.upper().startswith("PASS"):
        return QualityResult(True)
    reason = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict or "rejected"
    return QualityResult(False, reason)


def review(
    text: str,
    platform: str,
    limit: int,
    topic: str,
    editor: TextProvider | None = None,
    banned_words: list[str] | None = None,
) -> QualityResult:
    """Run the rule layer, then (if it passes and an editor is given) the AI editor."""
    rules = check_rules(text, platform, limit, banned_words)
    if not rules.passed:
        return rules
    if editor is None:
        return rules
    return ai_review(editor, text, platform, topic)
