"""Common interface every image provider must implement."""
from abc import ABC, abstractmethod


class ImageProvider(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return a URL (or local path) to the generated image."""
        raise NotImplementedError
