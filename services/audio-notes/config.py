#!/usr/bin/env python3
"""
Configuration for Audio Notes app.
"""

import logging
import os
from pathlib import Path

# Service URLs - Required: set in docker-compose.yaml for Docker mode
WHISPER_URL = os.environ.get("WHISPER_URL", "http://localhost:8003")
PARAKEET_URL = os.environ.get("PARAKEET_URL", "http://localhost:8002")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b")

# Recordings directory - uses Docker volume mount at /app/recordings
# The actual host path is configured in docker-compose.yaml
RECORDINGS_DIR = Path("/app/recordings")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
