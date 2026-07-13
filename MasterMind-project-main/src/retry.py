"""Shared retry decorator with exponential backoff."""
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

log = logging.getLogger("retry")


def with_retry(max_attempts: int = 3):
    """Retry on any exception with exponential backoff (2s, 4s, 8s...)."""
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(log, logging.WARNING),
    )
