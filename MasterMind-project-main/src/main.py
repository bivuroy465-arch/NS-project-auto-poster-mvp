"""Orchestrator: runs the full daily flow.

Flow: validate -> read state -> pick topic -> image -> write -> publish -> log.
Run locally:  python -m src.main
Dry run:      DRY_RUN=true python -m src.main   (generates content, no publishing)

This module is, and stays, fully synchronous - `asyncio` is never imported
here. `runtime.start_runtime()` starts a background event loop (owned by a
`LifecycleManager`) hosting the semantic duplicate guard and the
self-learning feedback loop, and returns `AppRuntime`, a plain object whose
methods happen to be backed by that loop. The background loop is started
once, up front, and always torn down in a `finally` block before `run()`
returns - cleanly, whether the run succeeded, failed, or raised.
"""

import sys
from typing import cast

from . import notifier, sheets_logger
from .config_loader import load_config
from .content_writer import produce_post
from .event_bus import Platform, PostPublished
from .exceptions import AutoPosterError
from .image_generator import generate_image
from .logging_setup import get_logger
from .providers.image.factory import get_image_chain
from .providers.text.factory import get_text_provider
from .publishers.factory import get_publisher
from .runtime import AppRuntime, start_runtime
from .topic_generator import generate_topic

log = get_logger("main")


def run() -> int:
    cfg = load_config()
    dry_run = bool(cfg.get("dry_run"))
    if dry_run:
        log.warning("DRY RUN enabled - content will be generated but NOT published.")

    text = get_text_provider(cfg["text_provider"], cfg["text_model"])
    image_chain = get_image_chain(cfg)

    # Optional AI editor for the quality gate.
    qg = cfg.get("quality_gate", {}) or {}
    editor = None
    if qg.get("enabled"):
        editor = get_text_provider(
            qg.get("editor_provider", cfg["text_provider"]),
            qg.get("editor_model", cfg["text_model"]),
        )

    recent = sheets_logger.recent_topics(limit=cfg.get("recent_topics_window", 30))
    log.info("Loaded %d recent topics for duplicate guard.", len(recent))

    # Background event loop for the semantic dedup guard and the
    # self-learning feedback loop. Started once, up front, and always torn
    # down before returning - everything below this point is still plain
    # synchronous code calling plain synchronous methods on `runtime`.
    runtime = start_runtime(cfg)
    try:
        try:
            runtime.hydrate_topics_sync(recent)
        except Exception as exc:
            log.warning(
                "Semantic dedup hydration failed (%s); continuing with an empty index.", exc
            )

        topic = _select_topic(text, cfg, recent, runtime)
        log.info("Topic chosen: %s", topic)

        # Best-effort: generate_image never raises and returns "" if all providers fail.
        image_url = generate_image(image_chain, topic, cfg.get("image_style", "anime"))
        if image_url:
            log.info("Image ready: %s", image_url)
        else:
            log.warning("No image available; continuing text-only.")

        limits = cfg.get("limits", {})
        failures = 0
        for platform in cfg["platforms"]:
            post = ""
            try:
                exemplars = runtime.get_few_shot_exemplars_sync(platform)
                post = produce_post(
                    text,
                    platform,
                    topic,
                    limits.get(platform, 1000),
                    editor=editor,
                    banned_words=qg.get("banned_words"),
                    max_attempts=qg.get("max_attempts", cfg.get("max_retries", 3)),
                    exemplars=exemplars,
                )
                if post is None:  # circuit breaker tripped; already alerted
                    failures += 1
                    status = "skipped:quality_gate"
                elif dry_run:
                    status = "dry_run"
                    log.info("[%s] DRY RUN post:\n%s", platform, post)
                else:
                    result = get_publisher(platform)(post, image_url)
                    status = f"published:{result.get('id', '')}"
                    log.info("[%s] published %s", platform, status)
                    _notify_feedback_loop(runtime, platform, topic, post, image_url, result)
            except Exception as exc:  # isolate: one platform must not stop others
                failures += 1
                status = f"error:{exc}"
                log.error("[%s] failed: %s", platform, exc)
            try:
                sheets_logger.log_row(topic, platform, post or "", image_url, status)
            except Exception as log_exc:
                log.error("[%s] sheet log failed: %s", platform, log_exc)

        if failures:
            log.error("%d/%d platforms failed", failures, len(cfg["platforms"]))
            notifier.send_alert(
                f"{failures}/{len(cfg['platforms'])} platform(s) failed "
                f"for topic: {topic!r}. Check the pipeline logs."
            )
            return 1
        log.info("All platforms processed successfully.")
        return 0
    finally:
        try:
            runtime.shutdown()
        except Exception as exc:
            log.error("Error shutting down the background runtime: %s", exc)


def _select_topic(text, cfg: dict, recent: list[str], runtime: AppRuntime) -> str:
    """Pick a topic that passes both the lexical guard (topic_generator,
    fast/exact-repeat-only) and the semantic guard (SemanticDeduplicator,
    catches paraphrases), retrying a bounded number of times if the
    semantic guard flags something the lexical guard missed.

    Fails open at every step: a broken/unavailable semantic guard degrades
    to lexical-only duplicate checking (what topic_generator already did on
    its own) rather than ever crashing topic selection.
    """
    sd_cfg = cfg.get("semantic_dedup", {}) or {}
    themes = cfg.get("topic_themes")
    max_retries = cfg.get("max_retries", 3)

    def _generate() -> str:
        return generate_topic(text, themes=themes, recent=recent, max_attempts=max_retries)

    if not sd_cfg.get("enabled", True):
        return _generate()

    attempts = sd_cfg.get("max_attempts", 2)
    topic = ""
    for attempt in range(1, attempts + 1):
        topic = _generate()
        try:
            verdict = runtime.check_topic_sync(topic)
        except Exception as exc:
            log.warning("Semantic duplicate guard unavailable (%s); proceeding without it.", exc)
            return topic
        if not verdict.is_duplicate:
            return topic
        log.warning(
            "Topic %r flagged as a semantic duplicate on attempt %d/%d (method=%s, score=%s).",
            topic,
            attempt,
            attempts,
            verdict.method,
            verdict.score,
        )
    log.warning("Semantic duplicate guard exhausted %d attempts; using last candidate.", attempts)
    return topic


def _notify_feedback_loop(
    runtime: AppRuntime,
    platform: str,
    topic: str,
    post: str,
    image_url: str,
    result: dict,
) -> None:
    """Best-effort: the learning loop must never be able to fail a publish."""
    try:
        # cfg["platforms"] is already validated against exactly these three
        # values by config_loader.load_config(); the cast just tells the type
        # checker what that runtime validation already guarantees.
        runtime.publish_post_published_sync(
            PostPublished(
                platform=cast(Platform, platform),
                post_id=str(result.get("id", "")),
                topic=topic,
                text=post,
                image_url=image_url or "",
            )
        )
    except Exception as exc:
        log.warning("[%s] failed to notify the feedback loop: %s", platform, exc)


if __name__ == "__main__":
    try:
        sys.exit(run())
    except AutoPosterError as exc:
        log.critical("Fatal configuration/setup error: %s", exc)
        notifier.send_alert(f"Fatal configuration/setup error: {exc}")
        sys.exit(2)
