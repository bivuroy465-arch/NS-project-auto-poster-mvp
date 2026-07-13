"""Anthropic Claude text provider."""
import anthropic
from .base import TextProvider
from ...config_loader import env
from ...retry import with_retry
from ...exceptions import ProviderError
from ...logging_setup import get_logger

log = get_logger("provider.claude")


class AnthropicTextProvider(TextProvider):
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        super().__init__(model)
        self.client = anthropic.Anthropic(api_key=env("ANTHROPIC_API_KEY", required=True))

    @with_retry()
    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system or "You are a helpful social media content assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as exc:
            log.error("Claude text generation failed: %s", exc)
            raise ProviderError(str(exc)) from exc
