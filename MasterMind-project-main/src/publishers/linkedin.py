"""Publishes a post to LinkedIn via the UGC Posts API."""
import requests
from ..config_loader import env
from ..retry import with_retry
from ..exceptions import PublishError
from ..logging_setup import get_logger

log = get_logger("publisher.linkedin")
API = "https://api.linkedin.com/v2/ugcPosts"


@with_retry()
def publish(text: str, image_url: str | None = None) -> dict:
    try:
        token = env("LINKEDIN_ACCESS_TOKEN", required=True)
        author = env("LINKEDIN_AUTHOR_URN", required=True)  # e.g. urn:li:person:xxxx
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Restli-Protocol-Version": "2.0.0",
            "Content-Type": "application/json",
        }
        body = {
            "author": author,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": text},
                    "shareMediaCategory": "NONE",
                }
            },
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
        }
        resp = requests.post(API, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        post_id = resp.headers.get("x-restli-id", "")
        log.info("LinkedIn post published: %s", post_id)
        return {"platform": "linkedin", "id": post_id}
    except Exception as exc:
        log.error("LinkedIn publish failed: %s", exc)
        raise PublishError(str(exc)) from exc
