"""Logs every generated post to a Google Sheet and reads recent topics (state)."""

import datetime as dt
import json

import gspread
from google.oauth2.service_account import Credentials

from .config_loader import env, env_present
from .logging_setup import get_logger

log = get_logger("sheets")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
_TOPIC_COL = 2  # 1-based column index of the topic (timestamp=1, topic=2, ...)


def is_configured() -> bool:
    return env_present("GOOGLE_SERVICE_ACCOUNT_JSON", "GOOGLE_SHEET_ID")


def _client() -> gspread.Client:
    raw = env("GOOGLE_SERVICE_ACCOUNT_JSON", required=True)
    creds = Credentials.from_service_account_info(json.loads(raw), scopes=SCOPES)
    return gspread.authorize(creds)


def _spreadsheet() -> gspread.Spreadsheet:
    return _client().open_by_key(env("GOOGLE_SHEET_ID", required=True))


def _sheet():
    return _spreadsheet().sheet1


def recent_topics(limit: int = 30) -> list[str]:
    """Return the most recent topics from the sheet (best-effort, newest first).

    Returns an empty list if Sheets is not configured or any error occurs,
    so topic generation never breaks because of state lookup.
    """
    if not is_configured():
        log.info("Sheets not configured; no recent topics for duplicate guard.")
        return []
    try:
        col = _sheet().col_values(_TOPIC_COL)
        # Drop a header cell if present, take the newest `limit`, newest first.
        values = [c.strip() for c in col if c.strip()]
        if values and values[0].lower() in ("topic", "topics"):
            values = values[1:]
        return list(reversed(values))[:limit]
    except Exception as exc:  # never let state lookup break the run
        log.warning("Could not read recent topics: %s", exc)
        return []


def log_row(topic: str, platform: str, text: str, image_url: str, status: str) -> None:
    if not is_configured():
        log.warning("Google Sheets not configured; skipping log for %s", platform)
        return
    _sheet().append_row(
        [
            dt.datetime.now(dt.UTC).isoformat(),
            topic,
            platform,
            text,
            image_url or "",
            status,
        ]
    )
    log.info("Logged %s row to Google Sheets", platform)


# --------------------------------------------------------------------------
# Durable state for the self-learning feedback loop (feedback_loop.py).
#
# A dedicated worksheet tab, separate from the main post log above: its
# shape (one row per engagement *reading*, not per publish attempt) and
# lifecycle (read back and ranked by SheetsMemoryRepository) are different
# enough from log_row/recent_topics to warrant their own tab rather than
# overloading sheet1's columns.
# --------------------------------------------------------------------------
_ENGAGEMENT_SHEET_TITLE = "Engagement"
_ENGAGEMENT_HEADERS = ["collected_at", "platform", "topic", "text", "engagement_rate"]


def _engagement_sheet():
    """Return the 'Engagement' worksheet, creating it (with headers) if missing."""
    spreadsheet = _spreadsheet()
    try:
        return spreadsheet.worksheet(_ENGAGEMENT_SHEET_TITLE)
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(
            title=_ENGAGEMENT_SHEET_TITLE,
            rows=2000,
            cols=len(_ENGAGEMENT_HEADERS),
        )
        sheet.append_row(_ENGAGEMENT_HEADERS)
        return sheet


def log_engagement(
    *,
    collected_at: str,
    platform: str,
    topic: str,
    text: str,
    engagement_rate: float,
) -> None:
    """Append one exemplar record to the durable 'Engagement' worksheet.

    Best-effort and fails open, like recent_topics() below - the feedback
    loop must never break the posting pipeline because Sheets is slow,
    misconfigured, or briefly unavailable.
    """
    if not is_configured():
        log.warning("Google Sheets not configured; skipping engagement log for %s", platform)
        return
    try:
        _engagement_sheet().append_row([collected_at, platform, topic, text, engagement_rate])
        log.info("Logged engagement row for %s (rate=%.4f)", platform, engagement_rate)
    except Exception as exc:
        log.warning("Failed to log engagement row: %s", exc)


def recent_engagement(platform: str, limit: int = 200) -> list[dict]:
    """Return up to `limit` most recent engagement rows for `platform` (best-effort)."""
    if not is_configured():
        return []
    try:
        rows = _engagement_sheet().get_all_records()
    except Exception as exc:
        log.warning("Could not read engagement rows: %s", exc)
        return []
    matching = [r for r in rows if r.get("platform") == platform]
    return matching[-limit:]
