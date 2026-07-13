"""Publishes a post to Twitter/X using tweepy, with optional image."""
import os
import tempfile

import requests
import tweepy

from ..config_loader import env
from ..exceptions import PublishError
from ..logging_setup import get_logger
from ..retry import with_retry

log = get_logger("publisher.twitter")


def _resolve_image(image_ref: str | None) -> str | None:
    """Return a local file path for an image given a URL or path (or None).

    An empty string (the text-only fallback) yields None, so the tweet is
    published without media and no API formatting error occurs.
    """
    if not image_ref:
        return None
    if os.path.exists(image_ref):
        return image_ref
    # Download remote URL to a temp file.
    resp = requests.get(image_ref, timeout=60)
    resp.raise_for_status()
    path = os.path.join(tempfile.gettempdir(), "twitter_media.png")
    with open(path, "wb") as f:
        f.write(resp.content)
    return path


@with_retry()
def publish(text: str, image_url: str | None = None) -> dict:
    try:
        client = tweepy.Client(
            consumer_key=env("TWITTER_API_KEY", required=True),
            consumer_secret=env("TWITTER_API_SECRET", required=True),
            access_token=env("TWITTER_ACCESS_TOKEN", required=True),
            access_token_secret=env("TWITTER_ACCESS_SECRET", required=True),
        )
        media_ids = None
        local = _resolve_image(image_url)
        if local:
            auth = tweepy.OAuth1UserHandler(
                env("TWITTER_API_KEY"), env("TWITTER_API_SECRET"),
                env("TWITTER_ACCESS_TOKEN"), env("TWITTER_ACCESS_SECRET"),
            )
            api_v1 = tweepy.API(auth)
            media = api_v1.media_upload(local)
            media_ids = [media.media_id]
        resp = client.create_tweet(text=text, media_ids=media_ids)
        tweet_id = str(resp.data.get("id"))
        log.info("Tweet published: %s", tweet_id)
        return {"platform": "twitter", "id": tweet_id}
    except Exception as exc:
        log.error("Twitter publish failed: %s", exc)
        raise PublishError(str(exc)) from exc
