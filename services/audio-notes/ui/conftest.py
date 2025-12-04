"""Pytest configuration for audio-notes UI tests.

This conftest.py is loaded before test collection, allowing us to mock
gradio before the ui package's __init__.py imports it.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock gradio module FIRST, before any other imports
# This is necessary because ui/__init__.py imports from ui.main which imports gradio
gradio_mock = MagicMock()
sys.modules["gradio"] = gradio_mock
sys.modules["gr"] = gradio_mock

# Add the service directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
