"""Common interface every text provider must implement."""
from abc import ABC, abstractmethod


class TextProvider(ABC):
    """All text/LLM providers implement generate().

    Adding a new provider = create a new file implementing this class
    and register it in factory.py. No other code changes needed.
    """

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        """Return the model's text response for the given prompt."""
        raise NotImplementedError
