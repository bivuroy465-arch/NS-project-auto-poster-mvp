"""Smoke test: validates config and imports without needing API keys.

Run:  python -m tests.smoke_test
Exits non-zero on the first failure so CI fails fast.
"""

import importlib
import os
import sys

# Avoid requiring real env vars during import-time checks.
os.environ.setdefault("DRY_RUN", "true")

FAILURES = []


def check(label: str, fn) -> None:
    try:
        fn()
        print(f"  OK   {label}")
    except Exception as exc:  # noqa: BLE001 - smoke test reports all failures
        print(f"  FAIL {label}: {exc}")
        FAILURES.append(label)


def _import_all() -> None:
    modules = [
        "src.config_loader",
        "src.logging_setup",
        "src.exceptions",
        "src.retry",
        "src.topic_generator",
        "src.content_writer",
        "src.image_generator",
        "src.sheets_logger",
        "src.main",
        "src.providers.text.factory",
        "src.providers.image.factory",
        "src.publishers.factory",
        "src.publishers.twitter",
        "src.publishers.linkedin",
        "src.publishers.facebook",
    ]
    for m in modules:
        importlib.import_module(m)


def _validate_config() -> None:
    from src.config_loader import load_config

    load_config()  # raises ConfigError on any invalid/missing key


def _registries_nonempty() -> None:
    from src.providers.image.factory import _REGISTRY as image_reg
    from src.providers.text.factory import _REGISTRY as text_reg
    from src.publishers.factory import _REGISTRY as pub_reg

    assert text_reg, "no text providers registered"
    assert image_reg, "no image providers registered"
    assert pub_reg, "no publishers registered"


def _config_providers_known() -> None:
    """Ensure the providers/platforms in config.yaml are actually registered."""
    from src.config_loader import load_config
    from src.providers.image.factory import _REGISTRY as image_reg
    from src.providers.text.factory import _REGISTRY as text_reg
    from src.publishers.factory import _REGISTRY as pub_reg

    cfg = load_config()
    assert cfg["text_provider"] in text_reg, (
        f"text_provider '{cfg['text_provider']}' not registered"
    )
    # image_provider (old, single string) / image_providers (new, fallback
    # chain) are alternatives - see providers.image.factory.get_image_chain.
    image_names = cfg.get("image_providers") or [cfg.get("image_provider")]
    if isinstance(image_names, str):
        image_names = [image_names]
    for name in image_names:
        assert name in image_reg, f"image_provider '{name}' not registered"
    for p in cfg["platforms"]:
        assert p in pub_reg, f"platform '{p}' has no publisher"


def main() -> int:
    print("Running smoke tests...")
    check("import all modules", _import_all)
    check("config.yaml is valid", _validate_config)
    check("factories are non-empty", _registries_nonempty)
    check("config providers/platforms are registered", _config_providers_known)

    if FAILURES:
        print(f"\n{len(FAILURES)} smoke test(s) failed: {FAILURES}")
        return 1
    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
