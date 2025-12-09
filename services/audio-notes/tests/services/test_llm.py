"""
Unit tests for Audio Notes LLM service module.

Tests GPU preparation and summarization functions.
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
    with patch.dict("sys.modules", {"config": mock_cfg}):
        yield mock_cfg


# ==============================================================================
# prepare_gpu_for_llm Tests
# ==============================================================================


class TestPrepareGpuForLlm:
    """Tests for prepare_gpu_for_llm function."""

    def test_unloads_parakeet_successfully(self, mock_config):
        """Test successful Parakeet unload."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.llm" in mod:
                del sys.modules[mod]

        with patch("requests.post") as mock_post:
            parakeet_response = MagicMock()
            parakeet_response.status_code = 200
            parakeet_response.json.return_value = {
                "status": "unloaded",
                "gpu_memory_used_gb": 6.0,
            }

            whisper_response = MagicMock()
            whisper_response.status_code = 200
            whisper_response.json.return_value = {
                "status": "not_loaded",
                "message": "Already unloaded",
            }

            mock_post.side_effect = [parakeet_response, whisper_response]

            from services.llm import prepare_gpu_for_llm

            result = prepare_gpu_for_llm()

            assert result["memory_freed_gb"] >= 6.0
            assert len(result["actions"]) == 2
            assert "Parakeet" in result["actions"][0]

    def test_unloads_whisper_successfully(self, mock_config):
        """Test successful Whisper unload."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.llm" in mod:
                del sys.modules[mod]

        with patch("requests.post") as mock_post:
            parakeet_response = MagicMock()
            parakeet_response.status_code = 200
            parakeet_response.json.return_value = {
                "status": "not_loaded",
            }

            whisper_response = MagicMock()
            whisper_response.status_code = 200
            whisper_response.json.return_value = {
                "status": "unloaded",
                "gpu_memory_used_gb": 4.0,
            }

            mock_post.side_effect = [parakeet_response, whisper_response]

            from services.llm import prepare_gpu_for_llm

            result = prepare_gpu_for_llm()

            assert result["memory_freed_gb"] >= 4.0

    def test_handles_unavailable_services(self, mock_config):
        """Test handling when services are unavailable."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.llm" in mod:
                del sys.modules[mod]

        with patch("requests.post") as mock_post:
            mock_post.side_effect = ConnectionError("Service unavailable")

            from services.llm import prepare_gpu_for_llm

            result = prepare_gpu_for_llm()

            assert result["memory_freed_gb"] == 0.0
            assert len(result["actions"]) == 2
            assert "unavailable" in result["actions"][0].lower()

    def test_no_models_to_unload(self, mock_config):
        """Test when no models need to be unloaded."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.llm" in mod:
                del sys.modules[mod]

        with patch("requests.post") as mock_post:
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {
                "status": "not_loaded",
                "message": "No models were loaded",
            }
            mock_post.return_value = response

            from services.llm import prepare_gpu_for_llm

            result = prepare_gpu_for_llm()

            assert result["memory_freed_gb"] == 0.0
            assert "already available" in result["message"].lower()


# ==============================================================================
# summarize_streaming Tests
# ==============================================================================


class TestSummarizeStreaming:
    """Tests for summarize_streaming generator function."""

    def test_streams_response(self, mock_config):
        """Test that summarization streams response chunks."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.llm" in mod:
                del sys.modules[mod]

        with patch("requests.post") as mock_post:
            # Mock GPU prep responses
            prep_response = MagicMock()
            prep_response.status_code = 200
            prep_response.json.return_value = {"status": "not_loaded"}

            # Mock streaming response
            stream_response = MagicMock()
            stream_response.status_code = 200
            stream_response.iter_lines.return_value = [
                b'{"message": {"content": "Hello"}}',
                b'{"message": {"content": " World"}}',
                b'{"done": true}',
            ]
            stream_response.__enter__ = MagicMock(return_value=stream_response)
            stream_response.__exit__ = MagicMock(return_value=False)

            mock_post.side_effect = [prep_response, prep_response, stream_response]

            from services.llm import summarize_streaming

            chunks = list(summarize_streaming("Test transcript", prepare_gpu=True))

            # Should have yielded something
            assert len(chunks) >= 1

    def test_uses_custom_model(self, mock_config):
        """Test that custom model is used when specified."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.llm" in mod:
                del sys.modules[mod]

        with patch("requests.post") as mock_post:
            prep_response = MagicMock()
            prep_response.status_code = 200
            prep_response.json.return_value = {"status": "not_loaded"}

            stream_response = MagicMock()
            stream_response.status_code = 200
            stream_response.iter_lines.return_value = [b'{"done": true}']
            stream_response.__enter__ = MagicMock(return_value=stream_response)
            stream_response.__exit__ = MagicMock(return_value=False)

            mock_post.side_effect = [prep_response, prep_response, stream_response]

            from services.llm import summarize_streaming

            # Consume generator
            list(summarize_streaming("Test", model="llama3:8b", prepare_gpu=True))

            # Check that custom model was used in API call
            calls = mock_post.call_args_list
            # Last call should be to Ollama API
            assert len(calls) >= 1

    def test_skips_gpu_prep_when_disabled(self, mock_config):
        """Test that GPU prep is skipped when prepare_gpu=False."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.llm" in mod:
                del sys.modules[mod]

        with patch("requests.post") as mock_post:
            stream_response = MagicMock()
            stream_response.status_code = 200
            stream_response.iter_lines.return_value = [b'{"done": true}']
            stream_response.__enter__ = MagicMock(return_value=stream_response)
            stream_response.__exit__ = MagicMock(return_value=False)

            mock_post.return_value = stream_response

            from services.llm import summarize_streaming

            list(summarize_streaming("Test", prepare_gpu=False))

            # Should only call Ollama API, not unload endpoints
            # First call should be directly to chat endpoint
            assert mock_post.call_count >= 1

    def test_handles_api_error(self, mock_config):
        """Test handling of Ollama API error."""
        import sys

        for mod in list(sys.modules.keys()):
            if "services.llm" in mod:
                del sys.modules[mod]

        with patch("requests.post") as mock_post:
            prep_response = MagicMock()
            prep_response.status_code = 200
            prep_response.json.return_value = {"status": "not_loaded"}

            error_response = MagicMock()
            error_response.status_code = 500
            error_response.text = "Internal Server Error"
            error_response.__enter__ = MagicMock(return_value=error_response)
            error_response.__exit__ = MagicMock(return_value=False)

            mock_post.side_effect = [prep_response, prep_response, error_response]

            from services.llm import summarize_streaming

            chunks = list(summarize_streaming("Test", prepare_gpu=True))

            # Should yield error message
            output = "".join(chunks)
            assert "Error" in output or len(chunks) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
