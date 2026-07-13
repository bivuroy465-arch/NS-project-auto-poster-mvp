"""Publishes a post to a Facebook Page via the Graph API."""
import os
import requests
from ..config_loader import env
from ..retry import with_retry
from ..exceptions import PublishError
from ..logging_setup import get_logger

log = get_logger("publisher.facebook")
GRAPH = "https://graph.facebook.com/v20.0"


@with_retry()
def publish(text: str, image_url: str | None = None) -> dict:
    try:
        page_id = env("FACEBOOK_PAGE_ID", required=True)
        token = env("FACEBOOK_PAGE_ACCESS_TOKEN", required=True)
        # Only remote URLs can be passed by reference; local files are uploaded.
        if image_url and image_url.startswith("http"):
            url = f"{GRAPH}/{page_id}/photos"
            data = {"url": image_url, "caption": text, "access_token": token}
            resp = requests.post(url, data=data, timeout=60)
        elif image_url and os.path.exists(image_url):
            url = f"{GRAPH}/{page_id}/photos"
            with open(image_url, "rb") as f:
                resp = requests.post(
                    url,
                    data={"caption": text, "access_token": token},
                    files={"source": f},
                    timeout=120,
                )
        else:
            url = f"{GRAPH}/{page_id}/feed"
            resp = requests.post(
                url, data={"message": text, "access_token": token}, timeout=60
            )
        resp.raise_for_status()
        post_id = resp.json().get("id", "")
        log.info("Facebook post published: %s", post_id)
        return {"platform": "facebook", "id": post_id}
    except Exception as exc:
        log.error("Facebook publish failed: %s", exc)
        raise PublishError(str(exc)) from exc
