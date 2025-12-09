"""Pytest configuration for whisper service tests."""

import sys
from pathlib import Path

# Add the service directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
