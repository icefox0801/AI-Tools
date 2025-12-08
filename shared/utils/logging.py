"""
Shared logging utilities for AI-Tools services.

Provides consistent logging configuration across all services.
"""

import logging
import os
import sys
from typing import Literal

# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Log level type
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(
    name: str | None = None,
    level: LogLevel | None = None,
    format: str = DEFAULT_FORMAT,
) -> logging.Logger:
    """
    Configure logging and return a logger.

    Args:
        name: Logger name (typically __name__). If None, returns root logger.
        level: Log level. Defaults to LOG_LEVEL env var or INFO.
        format: Log format string.

    Returns:
        Configured logger instance.

    Usage:
        from shared.utils import setup_logging
        logger = setup_logging(__name__)
        logger.info("Service started")
    """
    # Get level from env or default to INFO
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    log_level = getattr(logging, level, logging.INFO)

    # Configure root logger (only once)
    logging.basicConfig(
        level=log_level,
        format=format,
        stream=sys.stdout,
    )

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger by name.

    Assumes setup_logging() has been called at app startup.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


def set_log_level(level: LogLevel) -> None:
    """
    Change the log level at runtime.

    Args:
        level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
