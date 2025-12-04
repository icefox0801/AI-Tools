"""
Unit tests for Audio Notes health check module.

Tests health check functions for ASR and LLM services.
"""

from unittest.mock import MagicMock, patch

import pytest


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_config():
    """Mock config module."""
    mock_cfg = MagicMock()
    mock_cfg.PARAKEET_URL = "http://parakeet:8000"
    mock_cfg.WHISPER_URL = "http://whisper:8000"
    mock_cfg.OLLAMA_URL = "http://ollama:11434"
    mock_cfg.OLLAMA_MODEL = "qwen3:14b"
    mock_cfg.logger = MagicMock()
    with patch.dict("sys.modules", {"config": mock_cfg}):
        yield mock_cfg


# ==============================================================================
# check_whisper_health Tests
# ==============================================================================


class TestCheckWhisperHealth:
    """Tests for check_whisper_health function."""

    def test_whisper_healthy(self, mock_config):
        """Test Whisper health check returns healthy."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.health" in mod:
                del sys.modules[mod]

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            from services.health import check_whisper_health

            healthy, message = check_whisper_health()

            assert healthy is True
            assert "ready" in message.lower()

    def test_whisper_unhealthy_status(self, mock_config):
        """Test Whisper health check with bad status."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.health" in mod:
                del sys.modules[mod]

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_get.return_value = mock_response

            from services.health import check_whisper_health

            healthy, message = check_whisper_health()

            assert healthy is False
            assert "500" in message

    def test_whisper_connection_error(self, mock_config):
        """Test Whisper health check with connection error."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.health" in mod:
                del sys.modules[mod]

        with patch("requests.get") as mock_get:
            mock_get.side_effect = ConnectionError("Connection refused")

            from services.health import check_whisper_health

            healthy, message = check_whisper_health()

            assert healthy is False
            assert "not available" in message.lower()


# ==============================================================================
# check_parakeet_health Tests
# ==============================================================================


class TestCheckParakeetHealth:
    """Tests for check_parakeet_health function."""

    def test_parakeet_healthy(self, mock_config):
        """Test Parakeet health check returns healthy."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.health" in mod:
                del sys.modules[mod]

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            from services.health import check_parakeet_health

            healthy, message = check_parakeet_health()

            assert healthy is True
            assert "ready" in message.lower()

    def test_parakeet_unhealthy(self, mock_config):
        """Test Parakeet health check with bad status."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.health" in mod:
                del sys.modules[mod]

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_get.return_value = mock_response

            from services.health import check_parakeet_health

            healthy, message = check_parakeet_health()

            assert healthy is False

    def test_parakeet_timeout(self, mock_config):
        """Test Parakeet health check with timeout."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.health" in mod:
                del sys.modules[mod]

        with patch("requests.get") as mock_get:
            from requests.exceptions import Timeout

            mock_get.side_effect = Timeout("Request timed out")

            from services.health import check_parakeet_health

            healthy, message = check_parakeet_health()

            assert healthy is False


# ==============================================================================
# check_ollama_health Tests
# ==============================================================================


class TestCheckOllamaHealth:
    """Tests for check_ollama_health function."""

    def test_ollama_healthy_with_model(self, mock_config):
        """Test Ollama health check with model available."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.health" in mod:
                del sys.modules[mod]

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"name": "qwen3:14b"},
                    {"name": "llama3:8b"},
                ]
            }
            mock_get.return_value = mock_response

            from services.health import check_ollama_health

            healthy, message = check_ollama_health()

            assert healthy is True
            assert "qwen3:14b" in message

    def test_ollama_model_not_found(self, mock_config):
        """Test Ollama health when model not found."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.health" in mod:
                del sys.modules[mod]

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"name": "llama3:8b"},
                    {"name": "mistral:7b"},
                ]
            }
            mock_get.return_value = mock_response

            from services.health import check_ollama_health

            healthy, message = check_ollama_health()

            assert healthy is False
            assert "not found" in message.lower()

    def test_ollama_connection_error(self, mock_config):
        """Test Ollama health with connection error."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.health" in mod:
                del sys.modules[mod]

        with patch("requests.get") as mock_get:
            mock_get.side_effect = ConnectionError("Connection refused")

            from services.health import check_ollama_health

            healthy, message = check_ollama_health()

            assert healthy is False
            assert "not available" in message.lower()


# ==============================================================================
# get_ollama_models Tests
# ==============================================================================


class TestGetOllamaModels:
    """Tests for get_ollama_models function."""

    def test_get_models_success(self, mock_config):
        """Test getting available Ollama models."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.health" in mod:
                del sys.modules[mod]

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"name": "llama3:8b"},
                    {"name": "qwen3:14b"},
                    {"name": "mistral:7b"},
                ]
            }
            mock_get.return_value = mock_response

            from services.health import get_ollama_models

            models = get_ollama_models()

            assert len(models) == 3
            # Default model should be first
            assert models[0] == "qwen3:14b"

    def test_get_models_default_on_error(self, mock_config):
        """Test default model returned on error."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.health" in mod:
                del sys.modules[mod]

        with patch("requests.get") as mock_get:
            mock_get.side_effect = ConnectionError("Connection refused")

            from services.health import get_ollama_models

            models = get_ollama_models()

            assert len(models) == 1
            assert "qwen3:14b" in models[0]

    def test_get_models_empty_list(self, mock_config):
        """Test handling of empty models list."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.health" in mod:
                del sys.modules[mod]

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"models": []}
            mock_get.return_value = mock_response

            from services.health import get_ollama_models

            models = get_ollama_models()

            # Should return default model when list is empty
            assert len(models) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
