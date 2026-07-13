"""Returns configured image provider(s) by name."""
from ...exceptions import ConfigError
from .base import ImageProvider
from .openai_image import OpenAIImageProvider
from .stability_image import StabilityImageProvider

_REGISTRY = {
    "openai": OpenAIImageProvider,
    "stability": StabilityImageProvider,
}


def get_image_provider(name: str, model: str) -> ImageProvider:
    name = name.lower()
    if name not in _REGISTRY:
        raise ConfigError(
            f"Unknown image_provider '{name}'. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name](model=model)


def get_image_chain(cfg: dict) -> list[ImageProvider]:
    """Build an ordered list of image providers from config.

    Backward compatible:
    - new style:  image_providers: [openai, stability]
    - old style:  image_provider: openai   (single string)
    `image_model` is applied to every provider in the chain.
    """
    model = cfg.get("image_model", "")
    names = cfg.get("image_providers")
    if not names:
        single = cfg.get("image_provider")
        names = [single] if single else []
    if isinstance(names, str):  # tolerate a single string in image_providers
        names = [names]
    if not names:
        raise ConfigError("No image provider configured (image_provider/image_providers).")
    return [get_image_provider(n, model) for n in names]
