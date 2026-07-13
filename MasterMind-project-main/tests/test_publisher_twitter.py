"""Boundary tests for the Twitter publisher.

tests/test_main_integration.py deliberately replaces `get_publisher` with a
fake, so it never runs the real code in src/publishers/twitter.py. That
leaves a coverage gap: nothing verifies that this module builds the tweepy
request correctly, uploads media, or translates SDK failures into
PublishError.

These tests close that gap by exercising the real `publish()` function and
mocking only the third-party SDK/HTTP layer (tweepy, requests) - no network
calls are made.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.exceptions import PublishError
from src.publishers import twitter


@pytest.fixture(autouse=True)
def _twitter_env(monkeypatch):
    monkeypatch.setenv("TWITTER_API_KEY", "key")
    monkeypatch.setenv("TWITTER_API_SECRET", "secret")
    monkeypatch.setenv("TWITTER_ACCESS_TOKEN", "token")
    monkeypatch.setenv("TWITTER_ACCESS_SECRET", "token-secret")


def test_publish_text_only_success():
    fake_client = MagicMock()
    fake_client.create_tweet.return_value = MagicMock(data={"id": "999"})

    with patch.object(twitter.tweepy, "Client", return_value=fake_client) as client_cls:
        result = twitter.publish("Ship small, ship often. #devtips", image_url=None)

    client_cls.assert_called_once_with(
        consumer_key="key",
        consumer_secret="secret",
        access_token="token",
        access_token_secret="token-secret",
    )
    fake_client.create_tweet.assert_called_once_with(
        text="Ship small, ship often. #devtips",
        media_ids=None,
    )
    assert result == {"platform": "twitter", "id": "999"}


def test_publish_with_image_uploads_media_first():
    fake_client = MagicMock()
    fake_client.create_tweet.return_value = MagicMock(data={"id": "1000"})

    fake_media = MagicMock(media_id="media-42")
    fake_api = MagicMock()
    fake_api.media_upload.return_value = fake_media

    fake_response = MagicMock()
    fake_response.content = b"fake-png-bytes"
    fake_response.raise_for_status.return_value = None

    with (
        patch.object(twitter.tweepy, "Client", return_value=fake_client),
        patch.object(twitter.tweepy, "OAuth1UserHandler", return_value=MagicMock()),
        patch.object(twitter.tweepy, "API", return_value=fake_api),
        patch.object(twitter.requests, "get", return_value=fake_response) as get_mock,
    ):
        result = twitter.publish("New post", image_url="https://img.example.com/pic.png")

    # No real network call: requests.get was intercepted.
    get_mock.assert_called_once_with("https://img.example.com/pic.png", timeout=60)
    fake_api.media_upload.assert_called_once()
    fake_client.create_tweet.assert_called_once_with(text="New post", media_ids=["media-42"])
    assert result == {"platform": "twitter", "id": "1000"}


@patch("time.sleep", return_value=None)  # neutralize tenacity's exponential backoff
def test_publish_failure_is_wrapped_and_retried(_sleep):
    fake_client = MagicMock()
    fake_client.create_tweet.side_effect = RuntimeError("Twitter API is down")

    with patch.object(twitter.tweepy, "Client", return_value=fake_client):
        with pytest.raises(PublishError):
            twitter.publish("Will fail", image_url=None)

    # with_retry() retries up to 3 attempts before re-raising as PublishError.
    assert fake_client.create_tweet.call_count == 3
