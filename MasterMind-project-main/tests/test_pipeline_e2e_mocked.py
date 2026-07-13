"""End-to-end pipeline test built with unittest.mock (patch/MagicMock).

Complements tests/test_main_integration.py, which uses pytest's `monkeypatch`
fixture. This file shows the equivalent style using `unittest.mock`, and adds
a scenario the existing suite doesn't cover yet: a *multi-platform* run where
one platform is skipped by the quality gate while a sibling platform still
publishes successfully. This proves platform failures are isolated from each
other, and that the pipeline's public "seams" (text/image providers,
publishers, sheets, notifier) are exercised without any real network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import src.main as main
from src.semantic_dedup import DedupMethod, DedupVerdict


def _fake_runtime() -> MagicMock:
    """MagicMock stand-in for runtime.AppRuntime, matching this file's
    unittest.mock style (see test_main_integration.py for the plain-class
    equivalent).
    """
    runtime = MagicMock()
    runtime.check_topic_sync.return_value = DedupVerdict(
        is_duplicate=False,
        method=DedupMethod.SEMANTIC_LOCAL,
        corpus_size=0,
        latency_ms=0.1,
    )
    runtime.get_few_shot_exemplars_sync.return_value = []
    return runtime


def _text_provider(topic: str, good_platforms: tuple[str, ...]):
    """Fake text provider whose post quality depends on the target platform."""
    provider = MagicMock()

    def _generate(prompt, system="", max_tokens=1024):
        if "Judge this post" in prompt:
            return "PASS"
        if "Pick ONE" in prompt:
            return topic
        if any(f"Write a {p} post" in prompt for p in good_platforms):
            return "Ship small, ship often. #devtips"
        # Triggers the AI-isms rule regardless of platform, so this branch
        # always fails the deterministic rule layer.
        return "In conclusion, this post will be rejected by the rule layer."

    provider.generate.side_effect = _generate
    return provider


def test_multiplatform_isolation_with_mocked_boundaries():
    """Linkedin is skipped by the quality gate; Twitter still publishes."""
    cfg = {
        "text_provider": "x",
        "text_model": "m",
        "image_providers": ["a"],
        "image_model": "m",
        "image_style": "anime",
        "platforms": ["twitter", "linkedin"],
        "limits": {"twitter": 280, "linkedin": 3000},
        "max_retries": 2,
        "recent_topics_window": 5,
        "quality_gate": {"enabled": True, "max_attempts": 2},
        "dry_run": False,
    }
    text_provider = _text_provider("Edge computing in 2026", good_platforms=("twitter",))
    publish_fn = MagicMock(return_value={"id": "123"})

    runtime = _fake_runtime()
    with (
        patch.object(main, "load_config", return_value=cfg),
        patch.object(main, "get_text_provider", return_value=text_provider),
        patch.object(main, "get_image_chain", return_value=["chain"]),
        patch.object(main, "generate_image", return_value="img-1"),
        patch.object(main, "get_publisher", return_value=publish_fn) as get_publisher,
        patch.object(main, "start_runtime", return_value=runtime),
        patch.object(main.sheets_logger, "recent_topics", return_value=[]),
        patch.object(main.sheets_logger, "log_row") as log_row,
        patch.object(main.notifier, "send_alert") as send_alert,
    ):
        rc = main.run()

    # One of two platforms failed -> non-zero exit code.
    assert rc == 1

    # Twitter was published with exactly the text/image the pipeline produced.
    get_publisher.assert_called_once_with("twitter")
    publish_fn.assert_called_once_with("Ship small, ship often. #devtips", "img-1")

    # Both platforms are logged to Sheets regardless of outcome.
    assert log_row.call_count == 2
    logged_platforms = {call.args[1] for call in log_row.call_args_list}
    assert logged_platforms == {"twitter", "linkedin"}

    # Two distinct alerts: the circuit breaker (content_writer) and the
    # orchestrator's final failure summary (main.run).
    messages = [call.args[0] for call in send_alert.call_args_list]
    assert any("Quality Gate rejected content" in m and "linkedin" in m for m in messages)
    assert any("1/2 platform(s) failed" in m for m in messages)

    # The background runtime is always started and shut down, and only the
    # platform that actually published notifies the feedback loop.
    runtime.shutdown.assert_called_once()
    runtime.publish_post_published_sync.assert_called_once()
    published_event = runtime.publish_post_published_sync.call_args[0][0]
    assert published_event.platform == "twitter"
