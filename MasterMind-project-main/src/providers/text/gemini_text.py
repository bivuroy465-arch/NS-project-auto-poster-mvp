"""Google Gemini text provider."""
import google.generativeai as genai
from .base import TextProvider
from ...config_loader import env
from ...retry import with_retry
from ...exceptions import ProviderError
from ...logging_setup import get_logger

log = get_logger("provider.gemini")


class GeminiTextProvider(TextProvider):
    def __init__(self, model: str = "gemini-1.5-flash"):
        super().__init__(model)
        genai.configure(api_key=env("GEMINI_API_KEY", required=True))
        self._model = genai.GenerativeModel(model)

    @with_retry()
    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        try:
            full = f"{system}\n\n{prompt}" if system else prompt
            resp = self._model.generate_content(
                full,
                generation_config={"max_output_tokens": max_tokens},
            )
            return resp.text.strip()
        except Exception as exc:
            log.error("Gemini text generation failed: %s", exc)
            raise ProviderError(str(exc)) from exc
