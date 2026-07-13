"""End-to-end mocked test of the orchestrator (master regression guard).

Exercises all three Phase 3 pillars together with the core flow, with no
network calls: topic -> image fallback -> quality gate -> publish -> log.
"""

import src.main as main
from src.semantic_dedup import DedupMethod, DedupVerdict


class _FakeRuntime:
    """Duck-typed stand-in for runtime.AppRuntime - no real event loop.

    main.py only ever calls these five methods on whatever start_runtime()
    returns, so a plain fake with matching methods is enough to exercise
    the orchestrator without pulling in asyncio/threading/faiss/fastembed.
    """

    def __init__(self, *, duplicate_topics=None, exemplars=None):
        self._duplicate_topics = set(duplicate_topics or ())
        self._exemplars = exemplars or []
        self.published_events = []
        self.hydrated_with = None
        self.shutdown_called = False
        self.check_calls = 0

    def hydrate_topics_sync(self, recent_topics, *, timeout=30.0):
        self.hydrated_with = list(recent_topics)

    def check_topic_sync(self, topic, *, timeout=20.0):
        self.check_calls += 1
        is_dup = topic in self._duplicate_topics
        return DedupVerdict(
            is_duplicate=is_dup,
            method=DedupMethod.SEMANTIC_LOCAL,
            corpus_size=0,
            latency_ms=0.1,
        )

    def get_few_shot_exemplars_sync(self, platform, k=None, *, timeout=5.0):
        return self._exemplars

    def publish_post_published_sync(self, event, *, timeout=5.0):
        self.published_events.append(event)

    def shutdown(self, *, timeout=10.0):
        self.shutdown_called = True


class _Text:
    """A text provider that always returns a clean, passing post/topic."""

    def generate(self, prompt, system="", max_tokens=1024):
        if "Judge this post" in prompt:  # AI editor verdict
            return "PASS"
        if "Pick ONE" in prompt:  # topic generation
            return "Edge computing in 2026"
        return "Ship small, ship often. #devtips"  # post body


def _base_cfg(**over):
    cfg = {
        "text_provider": "x",
        "text_model": "m",
        "image_providers": ["a"],
        "image_model": "m",
        "image_style": "anime",
        "platforms": ["twitter"],
        "limits": {"twitter": 280},
        "max_retries": 2,
        "recent_topics_window": 5,
        "quality_gate": {"enabled": True, "max_attempts": 2},
        "dry_run": False,
    }
    cfg.update(over)
    return cfg


def _wire(
    monkeypatch,
    cfg,
    *,
    image_ref="img-1",
    publish_ok=True,
    published=None,
    duplicate_topics=None,
    exemplars=None,
    recent=(),
):
    """Patch every external boundary of run() with in-memory fakes."""
    published = published if published is not None else []
    monkeypatch.setattr(main, "load_config", lambda: cfg)
    monkeypatch.setattr(main, "get_text_provider", lambda n, m: _Text())
    monkeypatch.setattr(main, "get_image_chain", lambda c: ["chain"])
    monkeypatch.setattr(main, "generate_image", lambda chain, topic, style="anime": image_ref)
    monkeypatch.setattr(main.sheets_logger, "recent_topics", lambda limit=30: list(recent))
    logged = []
    monkeypatch.setattr(
        main.sheets_logger,
        "log_row",
        lambda *a: logged.append(a),
    )
    alerts = []
    monkeypatch.setattr(main.notifier, "send_alert", lambda msg: alerts.append(msg))

    def _publisher(platform):
        def _pub(text, image):
            if not publish_ok:
                raise RuntimeError("publish boom")
            published.append((platform, text, image))
            return {"id": "123"}

        return _pub

    monkeypatch.setattr(main, "get_publisher", _publisher)

    runtime = _FakeRuntime(duplicate_topics=duplicate_topics, exemplars=exemplars)
    monkeypatch.setattr(main, "start_runtime", lambda cfg: runtime)
    return {"published": published, "logged": logged, "alerts": alerts, "runtime": runtime}


def test_happy_path_publishes_and_logs(monkeypatch):
    cfg = _base_cfg()
    state = _wire(monkeypatch, cfg)
    rc = main.run()
    assert rc == 0
    assert state["published"] == [("twitter", "Ship small, ship often. #devtips", "img-1")]
    assert state["alerts"] == []
    assert len(state["logged"]) == 1


def test_text_only_when_all_images_fail(monkeypatch):
    cfg = _base_cfg()
    state = _wire(monkeypatch, cfg, image_ref="")  # image fallback degraded
    rc = main.run()
    assert rc == 0
    # Published with empty image ref, no crash.
    assert state["published"][0][2] == ""


def test_publish_failure_alerts_and_returns_1(monkeypatch):
    cfg = _base_cfg()
    state = _wire(monkeypatch, cfg, publish_ok=False)
    rc = main.run()
    assert rc == 1
    assert any("platform(s) failed" in a for a in state["alerts"])


def test_quality_gate_skip_when_post_always_bad(monkeypatch):
    cfg = _base_cfg()
    state = _wire(monkeypatch, cfg)

    class _BadText:
        def generate(self, prompt, system="", max_tokens=1024):
            if "Pick ONE" in prompt:
                return "Edge computing in 2026"
            if "Judge this post" in prompt:
                return "PASS"
            return "no hashtag so twitter rule fails"  # always rejected

    monkeypatch.setattr(main, "get_text_provider", lambda n, m: _BadText())
    rc = main.run()
    assert rc == 1  # platform skipped counts as failure
    assert state["published"] == []  # nothing published
    assert any("Quality Gate rejected content" in a for a in state["alerts"])


def test_background_runtime_starts_hydrates_and_shuts_down(monkeypatch):
    """The background runtime is hydrated with recent topics up front and
    always shut down, even on a fully successful run.
    """
    cfg = _base_cfg()
    state = _wire(monkeypatch, cfg, recent=["old topic"])

    rc = main.run()

    assert rc == 0
    assert state["runtime"].hydrated_with == ["old topic"]
    assert state["runtime"].shutdown_called is True


def test_successful_publish_notifies_the_feedback_loop(monkeypatch):
    """Every successful publish must feed a PostPublished event to the
    learning loop - the whole point of Idea 1.
    """
    cfg = _base_cfg()
    state = _wire(monkeypatch, cfg)

    rc = main.run()

    assert rc == 0
    events = state["runtime"].published_events
    assert len(events) == 1
    assert events[0].platform == "twitter"
    assert events[0].post_id == "123"
    assert events[0].topic == "Edge computing in 2026"
    assert events[0].text == "Ship small, ship often. #devtips"


def test_dry_run_never_notifies_the_feedback_loop(monkeypatch):
    """dry_run doesn't actually publish, so nothing should be reported as published."""
    cfg = _base_cfg(dry_run=True)
    state = _wire(monkeypatch, cfg)

    rc = main.run()

    assert rc == 0
    assert state["runtime"].published_events == []


def test_semantic_duplicate_guard_retries_up_to_max_attempts(monkeypatch):
    """_Text always proposes the same topic string, and the fake guard is
    wired to always flag it as a duplicate - so main.py must exhaust the
    full `semantic_dedup.max_attempts` budget (proven by the call count),
    then fail open and publish with the last candidate rather than hang or
    crash the run.
    """
    cfg = _base_cfg(semantic_dedup={"enabled": True, "max_attempts": 3})
    state = _wire(monkeypatch, cfg, duplicate_topics={"Edge computing in 2026"})

    rc = main.run()

    assert rc == 0
    assert state["runtime"].check_calls == 3
    assert state["published"] != []  # fail-open: still publishes with the last candidate


def test_semantic_dedup_disabled_skips_the_semantic_check(monkeypatch):
    cfg = _base_cfg(semantic_dedup={"enabled": False})
    state = _wire(monkeypatch, cfg, duplicate_topics={"Edge computing in 2026"})

    rc = main.run()

    # Even though the fake guard would flag this topic, it's never
    # consulted, so the run proceeds and publishes normally.
    assert rc == 0
    assert state["published"] != []
