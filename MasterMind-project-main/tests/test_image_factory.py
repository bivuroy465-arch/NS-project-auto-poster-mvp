"""Tests for image provider chain construction (backward compatibility)."""
import pytest

from src.exceptions import ConfigError
from src.providers.image import factory


class _Dummy:
    def __init__(self, name):
        self.name = name


@pytest.fixture(autouse=True)
def _patch_registry(monkeypatch):
    monkeypatch.setattr(
        factory, "get_image_provider", lambda name, model: _Dummy(name)
    )


def test_new_style_list():
    chain = factory.get_image_chain(
        {"image_providers": ["openai", "stability"], "image_model": "m"}
    )
    assert [p.name for p in chain] == ["openai", "stability"]


def test_old_style_single_string():
    chain = factory.get_image_chain({"image_provider": "openai", "image_model": "m"})
    assert [p.name for p in chain] == ["openai"]


def test_string_in_image_providers_is_tolerated():
    chain = factory.get_image_chain({"image_providers": "stability"})
    assert [p.name for p in chain] == ["stability"]


def test_missing_config_raises():
    with pytest.raises(ConfigError):
        factory.get_image_chain({})
