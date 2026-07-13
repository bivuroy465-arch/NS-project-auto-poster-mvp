"""Unit tests for the image fallback chain."""
import src.image_generator as image_generator


class _OK:
    def __init__(self, ref):
        self._ref = ref

    def generate(self, prompt):
        return self._ref


class _Fail:
    def generate(self, prompt):
        raise RuntimeError("provider down")


def test_primary_success_no_alert(monkeypatch):
    alerts = []
    monkeypatch.setattr(image_generator.notifier, "send_alert", lambda m: alerts.append(m))
    out = image_generator.generate_image([_OK("img-1")], "AI")
    assert out == "img-1"
    assert alerts == []  # no fallback, no alert


def test_falls_back_to_second_provider(monkeypatch):
    alerts = []
    monkeypatch.setattr(image_generator.notifier, "send_alert", lambda m: alerts.append(m))
    out = image_generator.generate_image([_Fail(), _OK("img-2")], "AI")
    assert out == "img-2"
    assert len(alerts) == 1  # one fallback warning
    assert "falling back" in alerts[0]


def test_all_providers_fail_degrades_to_text_only(monkeypatch):
    alerts = []
    monkeypatch.setattr(image_generator.notifier, "send_alert", lambda m: alerts.append(m))
    out = image_generator.generate_image([_Fail(), _Fail()], "AI")
    assert out == ""  # graceful degradation
    # one fallback alert + one total-failure alert
    assert len(alerts) == 2
    assert "text-only" in alerts[-1]


def test_never_raises_on_failure(monkeypatch):
    monkeypatch.setattr(image_generator.notifier, "send_alert", lambda m: None)
    # Should return "" rather than propagate the exception.
    assert image_generator.generate_image([_Fail()], "AI") == ""
