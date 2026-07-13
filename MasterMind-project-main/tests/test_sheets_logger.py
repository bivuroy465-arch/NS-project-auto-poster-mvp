"""Tests for the engagement log added to sheets_logger.py.

Mocks at the same boundary as the rest of this codebase's Sheets-adjacent
tests would: `_spreadsheet()` is replaced with an in-memory fake, so no
real gspread/network/credential resolution ever runs.
"""

import gspread
import pytest

from src import sheets_logger


class _FakeWorksheet:
    def __init__(self):
        self.rows: list[list] = []

    def append_row(self, row):
        self.rows.append(row)

    def get_all_records(self):
        data_rows = [r for r in self.rows if r != sheets_logger._ENGAGEMENT_HEADERS]
        return [dict(zip(sheets_logger._ENGAGEMENT_HEADERS, row, strict=True)) for row in data_rows]


class _FakeSpreadsheet:
    def __init__(self):
        self._worksheets: dict[str, _FakeWorksheet] = {}

    def worksheet(self, title):
        if title not in self._worksheets:
            raise gspread.WorksheetNotFound(title)
        return self._worksheets[title]

    def add_worksheet(self, title, rows, cols):
        sheet = self._worksheets[title] = _FakeWorksheet()
        return sheet


@pytest.fixture(autouse=True)
def _sheets_configured(monkeypatch):
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", "{}")
    monkeypatch.setenv("GOOGLE_SHEET_ID", "sheet-id")


def test_log_engagement_creates_worksheet_with_headers_then_appends(monkeypatch):
    fake = _FakeSpreadsheet()
    monkeypatch.setattr(sheets_logger, "_spreadsheet", lambda: fake)

    sheets_logger.log_engagement(
        collected_at="2026-01-01T00:00:00+00:00",
        platform="twitter",
        topic="AI",
        text="hello #ai",
        engagement_rate=0.42,
    )

    sheet = fake._worksheets[sheets_logger._ENGAGEMENT_SHEET_TITLE]
    assert sheet.rows[0] == sheets_logger._ENGAGEMENT_HEADERS
    assert sheet.rows[1] == ["2026-01-01T00:00:00+00:00", "twitter", "AI", "hello #ai", 0.42]


def test_log_engagement_reuses_existing_worksheet(monkeypatch):
    fake = _FakeSpreadsheet()
    fake._worksheets[sheets_logger._ENGAGEMENT_SHEET_TITLE] = _FakeWorksheet()
    monkeypatch.setattr(sheets_logger, "_spreadsheet", lambda: fake)

    sheets_logger.log_engagement(
        collected_at="t",
        platform="twitter",
        topic="x",
        text="y",
        engagement_rate=0.1,
    )

    assert len(fake._worksheets) == 1  # no duplicate worksheet created


def test_recent_engagement_filters_by_platform_and_limit(monkeypatch):
    fake = _FakeSpreadsheet()
    monkeypatch.setattr(sheets_logger, "_spreadsheet", lambda: fake)

    for i in range(3):
        sheets_logger.log_engagement(
            collected_at=f"t{i}",
            platform="twitter",
            topic=f"topic-{i}",
            text="x",
            engagement_rate=0.1 * i,
        )
    sheets_logger.log_engagement(
        collected_at="t-other",
        platform="linkedin",
        topic="other",
        text="y",
        engagement_rate=0.9,
    )

    result = sheets_logger.recent_engagement("twitter", limit=2)
    assert [r["topic"] for r in result] == ["topic-1", "topic-2"]


def test_log_engagement_noop_when_not_configured(monkeypatch):
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.delenv("GOOGLE_SHEET_ID", raising=False)
    # Must not raise and must not attempt any network call.
    sheets_logger.log_engagement(
        collected_at="t", platform="twitter", topic="x", text="y", engagement_rate=0.1
    )


def test_recent_engagement_empty_when_not_configured(monkeypatch):
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.delenv("GOOGLE_SHEET_ID", raising=False)
    assert sheets_logger.recent_engagement("twitter") == []


def test_log_engagement_fails_open_on_sheets_error(monkeypatch):
    def _boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(sheets_logger, "_spreadsheet", _boom)
    # Must swallow the error, never raise.
    sheets_logger.log_engagement(
        collected_at="t", platform="twitter", topic="x", text="y", engagement_rate=0.1
    )


def test_recent_engagement_fails_open_on_sheets_error(monkeypatch):
    def _boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(sheets_logger, "_spreadsheet", _boom)
    assert sheets_logger.recent_engagement("twitter") == []
