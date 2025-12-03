#!/usr/bin/env python3
"""
Configuration for Audio Notes app.
"""

import os
import logging
from pathlib import Path

# Service URLs
WHISPER_URL = os.getenv("WHISPER_URL", "http://localhost:8003")
PARAKEET_URL = os.getenv("PARAKEET_URL", "http://localhost:8002")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")

# Recordings directory - uses Docker volume mount at /app/recordings
# The actual host path is configured in docker-compose.yaml
RECORDINGS_DIR = Path("/app/recordings")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
