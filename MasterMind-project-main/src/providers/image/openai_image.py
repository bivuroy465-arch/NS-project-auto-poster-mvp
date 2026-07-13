"""OpenAI DALL-E image provider."""
from openai import OpenAI
from .base import ImageProvider
from ...config_loader import env
from ...retry import with_retry
from ...exceptions import ProviderError
from ...logging_setup import get_logger

log = get_logger("provider.openai.image")


class OpenAIImageProvider(ImageProvider):
    def __init__(self, model: str = "dall-e-3"):
        super().__init__(model)
        self.client = OpenAI(api_key=env("OPENAI_API_KEY", required=True))

    @with_retry()
    def generate(self, prompt: str) -> str:
        try:
            resp = self.client.images.generate(
                model=self.model, prompt=prompt, size="1024x1024", n=1,
            )
            return resp.data[0].url
        except Exception as exc:
            log.error("DALL-E image generation failed: %s", exc)
            raise ProviderError(str(exc)) from exc
