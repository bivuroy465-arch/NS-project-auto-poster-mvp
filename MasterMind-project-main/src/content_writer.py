"""Writes platform-specific posts, with an optional quality gate + retries.

When few-shot exemplars from the self-learning feedback loop
(`feedback_loop.get_few_shot_exemplars`) are available, they are woven into
the SYSTEM prompt (not the user prompt): 'best' exemplars become explicit
stylistic targets to emulate, 'worst' exemplars become explicit
anti-patterns to strictly avoid. Keeping this in the system prompt mirrors
`quality_gate.ai_review`'s existing `system=`-based pattern - persona and
standing constraints belong there, while the per-call specifics (platform,
topic, character limit) stay in the user prompt.
"""

from collections.abc import Sequence

from . import notifier, quality_gate
from .feedback_loop import FewShotExemplar
from .logging_setup import get_logger
from .providers.text.base import TextProvider

log = get_logger("content")

_STYLE = {
    "twitter": "punchy and concise, 1-2 short sentences, up to 2 relevant hashtags",
    "linkedin": "professional and insightful, a few short paragraphs, value-driven",
    "facebook": "friendly and conversational, easy to read, lightly engaging",
}

_BASE_SYSTEM = (
    "You are a skilled social media copywriter. Write in a natural, human "
    "voice. Do NOT include any preamble, explanations, AI-isms, or "
    "placeholders. Return only the post text."
)


def _format_exemplars(exemplars: Sequence[FewShotExemplar] | None) -> str:
    """Render best/worst exemplars as explicit style guidance for the prompt.

    Best examples are framed as stylistic targets to emulate; worst
    examples are framed as anti-patterns to strictly avoid - both by
    explicit instruction, not just by showing the text and hoping the model
    infers the right lesson.
    """
    if not exemplars:
        return ""
    best = [e for e in exemplars if e.label == "best"]
    worst = [e for e in exemplars if e.label == "worst"]
    parts = []
    if best:
        examples = "\n".join(
            f'- "{e.text}" (engagement rate: {e.engagement_rate:.1%})' for e in best
        )
        parts.append(
            "These past posts performed well - emulate their tone, structure, and "
            f"hooks (write fresh content; never repeat them verbatim):\n{examples}"
        )
    if worst:
        examples = "\n".join(
            f'- "{e.text}" (engagement rate: {e.engagement_rate:.1%})' for e in worst
        )
        parts.append(
            "These past posts performed poorly - treat them as anti-patterns and "
            f"strictly avoid whatever made them fall flat:\n{examples}"
        )
    return "\n\n".join(parts)


def _build_system_prompt(exemplars: Sequence[FewShotExemplar] | None) -> str:
    guidance = _format_exemplars(exemplars)
    return f"{_BASE_SYSTEM}\n\n{guidance}" if guidance else _BASE_SYSTEM


def write_post(
    provider: TextProvider,
    platform: str,
    topic: str,
    limit: int,
    exemplars: Sequence[FewShotExemplar] | None = None,
) -> str:
    style = _STYLE.get(platform, "clear and engaging")
    prompt = (
        f"Write a {platform} post about: {topic}.\n"
        f"Tone/style: {style}.\n"
        f"Stay strictly under {limit} characters."
    )
    system = _build_system_prompt(exemplars)
    text = (provider.generate(prompt, system=system, max_tokens=800) or "").strip()
    return text[:limit]


def produce_post(
    provider: TextProvider,
    platform: str,
    topic: str,
    limit: int,
    editor: TextProvider | None = None,
    banned_words: list[str] | None = None,
    max_attempts: int = 3,
    exemplars: Sequence[FewShotExemplar] | None = None,
) -> str | None:
    """Write a post that passes the quality gate, regenerating on rejection.

    Circuit breaker: after `max_attempts` rejections, alert and return None so
    the caller can skip the platform without crashing the run.

    `exemplars` (typically from `AppRuntime.get_few_shot_exemplars_sync`) are
    passed through unchanged on every regeneration attempt - the style
    guidance they encode is a standing constraint, not something to drop on
    retry.
    """
    last_reason = ""
    for attempt in range(1, max_attempts + 1):
        text = write_post(provider, platform, topic, limit, exemplars=exemplars)
        result = quality_gate.review(
            text, platform, limit, topic, editor=editor, banned_words=banned_words
        )
        if result.passed:
            if attempt > 1:
                log.info("[%s] passed quality gate on attempt %d.", platform, attempt)
            return text
        last_reason = result.reason
        log.warning(
            "[%s] quality gate rejected attempt %d/%d: %s",
            platform,
            attempt,
            max_attempts,
            last_reason,
        )
    notifier.send_alert(
        f"Quality Gate rejected content {max_attempts} times for topic "
        f"{topic!r}. Last Reason: {last_reason}. Skipping platform {platform}."
    )
    log.error("[%s] skipped after %d quality-gate rejections.", platform, max_attempts)
    return None
