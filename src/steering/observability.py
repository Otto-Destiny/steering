"""Logging setup for the local daemon and CLI.

Capture failures are recoverable by design: a resolver falls back, a shortlink is
skipped, a provider is retried. Without logs those recoveries are invisible, and
a user who sees an empty result has no way to learn which step gave up. Every
such decision is logged, so `--log-level debug` explains an unexpected capture
instead of leaving the user to guess.
"""

from __future__ import annotations

import logging

PACKAGE_LOGGER = "steering"
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
LOG_LEVELS = ("critical", "error", "warning", "info", "debug")


def configure_logging(level: str = "info", *, stream: object | None = None) -> None:
    """Attach one stderr handler to the package logger, idempotently.

    Only the package logger is configured. The root logger is left alone so an
    application embedding STEERING keeps control of its own logging.
    """

    normalized = level.strip().lower()
    if normalized not in LOG_LEVELS:
        raise ValueError(f"log level must be one of {', '.join(LOG_LEVELS)}")

    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(normalized.upper())
    logger.propagate = False
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
    handler = logging.StreamHandler(stream)  # type: ignore[arg-type]
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
