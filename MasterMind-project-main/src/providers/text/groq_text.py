"""Groq text provider (fast, free tier available)."""
from groq import Groq
from .base import TextProvider
from ...config_loader import env
from ...retry import with_retry
from ...exceptions import ProviderError
from ...logging_setup import get_logger

log = get_logger("provider.groq")


class GroqTextProvider(TextProvider):
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        super().__init__(model)
        self.client = Groq(api_key=env("GROQ_API_KEY", required=True))

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
            log.error("Groq text generation failed: %s", exc)
            raise ProviderError(str(exc)) from exc
