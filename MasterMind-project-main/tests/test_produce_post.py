"""Tests for produce_post: regeneration, the circuit breaker, and the
few-shot exemplar guidance woven into the system prompt.
"""

import src.content_writer as content_writer
from src.feedback_loop import FewShotExemplar


class _Writer:
    """Provider that returns queued post bodies and records every call."""

    def __init__(self, posts):
        self._posts = list(posts)
        self.calls = 0
        self.seen_prompts = []
        self.seen_systems = []

    def generate(self, prompt, system="", max_tokens=1024):
        self.calls += 1
        self.seen_prompts.append(prompt)
        self.seen_systems.append(system)
        return self._posts.pop(0)


def test_returns_first_passing_post(monkeypatch):
    monkeypatch.setattr(content_writer.notifier, "send_alert", lambda m: None)
    writer = _Writer(["Ship small, ship often. #devtips"])
    out = content_writer.produce_post(writer, "twitter", "AI", 280, editor=None)
    assert out == "Ship small, ship often. #devtips"
    assert writer.calls == 1


def test_regenerates_until_rules_pass(monkeypatch):
    monkeypatch.setattr(content_writer.notifier, "send_alert", lambda m: None)
    # First post lacks a hashtag (fails Twitter rule), second passes.
    writer = _Writer(["no hashtag here", "now with a tag #ai"])
    out = content_writer.produce_post(writer, "twitter", "AI", 280, editor=None)
    assert out == "now with a tag #ai"
    assert writer.calls == 2


def test_circuit_breaker_alerts_and_returns_none(monkeypatch):
    alerts = []
    monkeypatch.setattr(content_writer.notifier, "send_alert", lambda m: alerts.append(m))
    # Every attempt fails the rule layer (no hashtag).
    writer = _Writer(["bad one", "bad two", "bad three"])
    out = content_writer.produce_post(writer, "twitter", "AI", 280, editor=None, max_attempts=3)
    assert out is None
    assert writer.calls == 3
    assert len(alerts) == 1
    assert "Skipping platform twitter" in alerts[0]
    assert "Quality Gate rejected content 3 times" in alerts[0]


def test_write_post_without_exemplars_uses_the_base_system_prompt_only():
    writer = _Writer(["Ship small, ship often. #devtips"])
    content_writer.write_post(writer, "twitter", "AI", 280)
    assert writer.seen_systems[0] == content_writer._BASE_SYSTEM


def test_write_post_frames_best_exemplars_as_targets_to_emulate():
    writer = _Writer(["Ship small, ship often. #devtips"])
    exemplars = [
        FewShotExemplar(label="best", topic="AI", text="Old great post #ai", engagement_rate=0.42),
    ]
    content_writer.write_post(writer, "twitter", "AI", 280, exemplars=exemplars)
    system = writer.seen_systems[0]
    assert "performed well" in system
    assert "emulate" in system
    assert "Old great post #ai" in system
    assert "performed poorly" not in system


def test_write_post_frames_worst_exemplars_as_anti_patterns_to_avoid():
    writer = _Writer(["Ship small, ship often. #devtips"])
    exemplars = [
        FewShotExemplar(label="worst", topic="AI", text="Old flop post", engagement_rate=0.01),
    ]
    content_writer.write_post(writer, "twitter", "AI", 280, exemplars=exemplars)
    system = writer.seen_systems[0]
    assert "performed poorly" in system
    assert "strictly avoid" in system
    assert "Old flop post" in system
    assert "performed well" not in system


def test_write_post_includes_both_best_and_worst_when_both_are_present():
    writer = _Writer(["Ship small, ship often. #devtips"])
    exemplars = [
        FewShotExemplar(label="best", topic="AI", text="Great post", engagement_rate=0.5),
        FewShotExemplar(label="worst", topic="AI", text="Flop post", engagement_rate=0.01),
    ]
    content_writer.write_post(writer, "twitter", "AI", 280, exemplars=exemplars)
    system = writer.seen_systems[0]
    assert "Great post" in system
    assert "Flop post" in system


def test_produce_post_forwards_exemplars_on_every_regeneration_attempt():
    writer = _Writer(["no hashtag here", "now with a tag #ai"])
    exemplars = [
        FewShotExemplar(label="best", topic="AI", text="Great post #ai", engagement_rate=0.5),
    ]
    content_writer.produce_post(writer, "twitter", "AI", 280, editor=None, exemplars=exemplars)
    assert writer.calls == 2
    assert all("Great post #ai" in system for system in writer.seen_systems)
