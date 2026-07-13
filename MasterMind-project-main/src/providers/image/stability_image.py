"""Stability AI image provider (good for anime-style)."""
import base64
import os
import tempfile
import requests
from .base import ImageProvider
from ...config_loader import env
from ...retry import with_retry
from ...exceptions import ProviderError
from ...logging_setup import get_logger

log = get_logger("provider.stability")
API = "https://api.stability.ai/v2beta/stable-image/generate/core"


class StabilityImageProvider(ImageProvider):
    """Returns a local file path to the generated PNG."""

    @with_retry()
    def generate(self, prompt: str) -> str:
        try:
            key = env("STABILITY_API_KEY", required=True)
            resp = requests.post(
                API,
                headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
                files={"none": ("", "")},
                data={"prompt": prompt, "output_format": "png"},
                timeout=120,
            )
            resp.raise_for_status()
            img_b64 = resp.json()["image"]
            path = os.path.join(tempfile.gettempdir(), "autopost_image.png")
            with open(path, "wb") as f:
                f.write(base64.b64decode(img_b64))
            return path
        except Exception as exc:
            log.error("Stability image generation failed: %s", exc)
            raise ProviderError(str(exc)) from exc
