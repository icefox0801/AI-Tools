"""Local logging setup for the video-upscaler service.

Kept self-contained so the service does not import the workspace `shared`
package (which eagerly pulls in httpx/langchain/ASR dependencies the upscaler
does not need).
"""

import logging
import os
import sys

DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(
    name: str | None = None,
    level: str | None = None,
    format: str = DEFAULT_FORMAT,
) -> logging.Logger:
    """Configure root logging once and return a named logger."""
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    log_level = getattr(logging, level, logging.INFO)

    logging.basicConfig(level=log_level, format=format, stream=sys.stdout)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    return logger
