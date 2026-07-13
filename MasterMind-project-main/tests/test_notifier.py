"""Unit tests for the fail-safe notifier."""
import src.notifier as notifier


def test_no_op_when_url_unset(monkeypatch):
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    assert notifier.is_configured() is False
    assert notifier.send_alert("hello") is False


def test_returns_false_and_does_not_raise_on_network_error(monkeypatch):
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://example.com/hook")

    def _boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(notifier.requests, "post", _boom)
    # Must swallow the error and return False, never raise.
    assert notifier.send_alert("hello") is False


def test_telegram_payload_shape(monkeypatch):
    url = "https://api.telegram.org/bot123/sendMessage"
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")
    payload = notifier._build_payload(url, "msg")
    assert payload == {"chat_id": "999", "text": "msg"}


def test_discord_payload_shape():
    url = "https://discord.com/api/webhooks/abc/def"
    assert notifier._build_payload(url, "msg") == {"content": "msg"}


def test_generic_payload_shape():
    assert notifier._build_payload("https://hooks.example.com/x", "msg") == {"text": "msg"}


def test_send_alert_success(monkeypatch):
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/x")
    sent = {}

    class _Resp:
        def raise_for_status(self):
            return None

    def _fake_post(url, json, timeout):
        sent["url"] = url
        sent["json"] = json
        return _Resp()

    monkeypatch.setattr(notifier.requests, "post", _fake_post)
    assert notifier.send_alert("hello") is True
    assert sent["url"] == "https://hooks.example.com/x"
    assert "hello" in sent["json"]["text"]
