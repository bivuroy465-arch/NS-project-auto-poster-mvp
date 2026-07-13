"""Unit tests for topic generation with the duplicate guard."""
from src.topic_generator import generate_topic


class _FakeProvider:
    """Returns queued responses in order to simulate the LLM."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def generate(self, prompt, system="", max_tokens=1024):
        self.calls += 1
        return self._responses.pop(0)


def test_returns_first_unique_topic():
    provider = _FakeProvider(["Quantum computing basics"])
    topic = generate_topic(provider, recent=["AI productivity"])
    assert topic == "Quantum computing basics"
    assert provider.calls == 1


def test_retries_on_duplicate_then_returns_unique():
    provider = _FakeProvider(["AI productivity", "Edge computing trends"])
    topic = generate_topic(provider, recent=["ai productivity"], max_attempts=3)
    assert topic == "Edge computing trends"
    assert provider.calls == 2


def test_gives_up_after_max_attempts_returns_last():
    provider = _FakeProvider(["AI productivity", "AI productivity", "AI productivity"])
    topic = generate_topic(provider, recent=["ai productivity"], max_attempts=3)
    assert topic == "AI productivity"
    assert provider.calls == 3
