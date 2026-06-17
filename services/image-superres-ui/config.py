"""Configuration for image super-resolution UI."""

import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("image-superres-ui")

IMAGE_SR_URL = os.environ.get("IMAGE_SR_URL", "http://image-superres:8000")
SERVER_NAME = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7862"))
