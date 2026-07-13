"""Unit tests for the decoupled quality gate (no publishers involved)."""
from src.quality_gate import ai_review, check_rules, review


class _Editor:
    """Returns queued verdicts to simulate the AI editor."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)

    def generate(self, prompt, system="", max_tokens=1024):
        return self._verdicts.pop(0)


# ---- rule layer ----

def test_rules_reject_empty():
    assert check_rules("  ", "twitter", 280).passed is False


def test_rules_reject_over_limit():
    res = check_rules("x" * 300, "twitter", 280)
    assert res.passed is False and "limit" in res.reason


def test_rules_reject_missing_hashtag_on_twitter():
    res = check_rules("A nice post with no tag", "twitter", 280)
    assert res.passed is False and "hashtag" in res.reason


def test_rules_reject_ai_ism():
    res = check_rules("Here is your post about AI #ai", "twitter", 280)
    assert res.passed is False and "AI-ism" in res.reason


def test_rules_reject_placeholder():
    res = check_rules("Check this out [insert link] #ai", "twitter", 280)
    assert res.passed is False and "placeholder" in res.reason


def test_rules_reject_banned_word():
    res = check_rules("Great crypto tips #ai", "twitter", 280, banned_words=["crypto"])
    assert res.passed is False and "banned" in res.reason


def test_rules_pass_clean_twitter_post():
    assert check_rules("Ship small, ship often. #devtips", "twitter", 280).passed is True


def test_rules_no_hashtag_required_for_linkedin():
    assert check_rules("A thoughtful professional update.", "linkedin", 3000).passed is True


# ---- AI editor layer ----

def test_ai_review_pass():
    assert ai_review(_Editor(["PASS"]), "clean post #ai", "twitter", "AI").passed is True


def test_ai_review_fail_with_reason():
    res = ai_review(_Editor(["FAIL: robotic tone"]), "meh", "twitter", "AI")
    assert res.passed is False and res.reason == "robotic tone"


# ---- combined review ----

def test_review_skips_ai_when_rules_fail():
    # Editor would PASS, but rules fail first (over limit) so AI is never used.
    res = review("x" * 300, "twitter", 280, "AI", editor=_Editor(["PASS"]))
    assert res.passed is False and "limit" in res.reason


def test_review_runs_ai_when_rules_pass():
    res = review("clean post #ai", "twitter", 280, "AI", editor=_Editor(["FAIL: off-topic"]))
    assert res.passed is False and res.reason == "off-topic"


def test_review_passes_without_editor():
    assert review("clean post #ai", "twitter", 280, "AI", editor=None).passed is True
