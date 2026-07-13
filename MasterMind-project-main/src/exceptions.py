"""Custom exceptions for clear, typed error handling."""


class AutoPosterError(Exception):
    """Base error for the application."""


class ConfigError(AutoPosterError):
    """Invalid or missing configuration."""


class ProviderError(AutoPosterError):
    """A text/image provider failed."""


class PublishError(AutoPosterError):
    """Publishing to a platform failed."""
