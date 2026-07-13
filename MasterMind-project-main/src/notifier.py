"""Fail-safe failure alerting via an outgoing webhook.

Design principles:
- Strictly opt-in: does nothing unless ALERT_WEBHOOK_URL is set.
- Never raises: a failure to alert must never break the main run.
- Supports Telegram bot API and Discord/Slack-style webhooks; the payload
  shape is chosen from the URL so no extra config is needed.

Environment:
- ALERT_WEBHOOK_URL   the full webhook URL (required to enable alerts)
- TELEGRAM_CHAT_ID    required only when using a Telegram bot URL
"""
import os

import requests

from .logging_setup import get_logger

log = get_logger("notifier")
_TIMEOUT = 10


def is_configured() -> bool:
    return bool(os.getenv("ALERT_WEBHOOK_URL"))


def _build_payload(url: str, message: str) -> dict:
    """Pick a payload shape based on the webhook provider."""
    if "api.telegram.org" in url:
        return {"chat_id": os.getenv("TELEGRAM_CHAT_ID", ""), "text": message}
    if "discord.com" in url or "discordapp.com" in url:
        return {"content": message}
    # Slack and most generic webhooks accept {"text": ...}.
    return {"text": message}


def send_alert(message: str) -> bool:
    """Send an alert. Returns True on success, False otherwise. Never raises."""
    url = os.getenv("ALERT_WEBHOOK_URL")
    if not url:
        log.info("ALERT_WEBHOOK_URL not set; skipping alert.")
        return False
    try:
        prefixed = f"\U0001F6A8 AI Auto-Poster: {message}"
        resp = requests.post(url, json=_build_payload(url, prefixed), timeout=_TIMEOUT)
        resp.raise_for_status()
        log.info("Alert sent.")
        return True
    except Exception as exc:  # alerting must never crash the run
        log.warning("Failed to send alert: %s", exc)
        return False
