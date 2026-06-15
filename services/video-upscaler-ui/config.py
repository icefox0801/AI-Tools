"""Configuration for the Video Upscaler Web UI."""

import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("video-upscaler-ui")

# Backend API (service name on the docker network)
UPSCALER_URL = os.environ.get("UPSCALER_URL", "http://video-upscaler:8000")

# UI server
SERVER_NAME = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7861"))

# How often to poll the backend for job progress (seconds)
POLL_INTERVAL_SEC = float(os.environ.get("POLL_INTERVAL_SEC", "2.0"))

# Request timeout for uploads (large videos)
UPLOAD_TIMEOUT_SEC = float(os.environ.get("UPLOAD_TIMEOUT_SEC", "600"))
