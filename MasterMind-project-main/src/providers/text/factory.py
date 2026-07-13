"""Returns the configured text provider by name."""
from .base import TextProvider
from .openai_text import OpenAITextProvider
from .anthropic_text import AnthropicTextProvider
from .gemini_text import GeminiTextProvider
from .groq_text import GroqTextProvider
from ...exceptions import ConfigError

# Register new providers here.
_REGISTRY = {
    "openai": OpenAITextProvider,
    "claude": AnthropicTextProvider,
    "gemini": GeminiTextProvider,
    "groq": GroqTextProvider,
}


def get_text_provider(name: str, model: str) -> TextProvider:
    name = name.lower()
    if name not in _REGISTRY:
        raise ConfigError(
            f"Unknown text_provider '{name}'. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name](model=model)
