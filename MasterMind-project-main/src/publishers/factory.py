"""Maps platform names to their publish() function."""
from . import twitter, linkedin, facebook

_REGISTRY = {
    "twitter": twitter.publish,
    "linkedin": linkedin.publish,
    "facebook": facebook.publish,
}


def get_publisher(platform: str):
    platform = platform.lower()
    if platform not in _REGISTRY:
        raise ValueError(f"Unknown platform '{platform}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[platform]
