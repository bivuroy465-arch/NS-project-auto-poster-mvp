"""OpenAI GPT text provider."""
from openai import OpenAI
from .base import TextProvider
from ...config_loader import env
from ...retry import with_retry
from ...exceptions import ProviderError
from ...logging_setup import get_logger

log = get_logger("provider.openai.text")


class OpenAITextProvider(TextProvider):
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__(model)
        self.client = OpenAI(api_key=env("OPENAI_API_KEY", required=True))

    @with_retry()
    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            log.error("OpenAI text generation failed: %s", exc)
            raise ProviderError(str(exc)) from exc
