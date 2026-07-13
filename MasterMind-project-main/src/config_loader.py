"""Loads config.yaml and environment variables, with validation."""

import os

import yaml
from dotenv import load_dotenv

from .exceptions import ConfigError

load_dotenv()  # loads .env locally; on CI the variables come from CI/CD settings

_REQUIRED_KEYS = ("text_provider", "text_model", "image_model", "platforms")
_VALID_PLATFORMS = {"twitter", "linkedin", "facebook"}


def load_config(path: str = "config.yaml") -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    missing = [k for k in _REQUIRED_KEYS if not cfg.get(k)]
    # image_provider (old, single string) / image_providers (new, fallback
    # chain) are alternatives - see providers.image.factory.get_image_chain,
    # which already accepts either. Require at least one, not both.
    if not cfg.get("image_provider") and not cfg.get("image_providers"):
        missing.append("image_provider(s)")
    if missing:
        raise ConfigError(f"Missing required config keys: {missing}")

    if not isinstance(cfg["platforms"], list) or not cfg["platforms"]:
        raise ConfigError("'platforms' must be a non-empty list")

    bad = set(cfg["platforms"]) - _VALID_PLATFORMS
    if bad:
        raise ConfigError(f"Unknown platforms {bad}. Valid: {_VALID_PLATFORMS}")

    # Environment can force dry-run regardless of file value.
    if os.getenv("DRY_RUN", "").lower() in ("1", "true", "yes"):
        cfg["dry_run"] = True
    return cfg


def env(name: str, required: bool = False, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if required and not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def env_present(*names: str) -> bool:
    """True only if every named variable is set and non-empty."""
    return all(os.getenv(n) for n in names)
