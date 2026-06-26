"""Structured logging configuration.

The original notebook communicated through ``print`` statements. This module
provides a single, idempotent entry point so library code can emit structured
logs while the notebook (or any caller) decides the verbosity.
"""

from __future__ import annotations

import logging

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
_configured = False


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure the root logger once.

    Safe to call repeatedly (e.g. from every notebook run); subsequent calls
    only adjust the level instead of stacking handlers.

    Args:
        level: Logging level as an ``int`` or its name (``"INFO"``, ``"DEBUG"``).
    """
    global _configured
    root = logging.getLogger()
    if not _configured:
        logging.basicConfig(level=level, format=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)
        _configured = True
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring logging on first use."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
