"""
Unit tests for shared.config.backends module.
"""

import pytest

from shared.config.backends import (
    BACKENDS_DOCKER,
    BACKENDS_LOCAL,
    format_model_name,
    get_backend_config,
    list_backends,
)


class TestBackendConfigs:
    """Tests for backend configuration dictionaries."""

    def test_backends_local_has_all_services(self):
        """Test BACKENDS_LOCAL contains all expected services."""
        expected = {"transcription", "vosk", "parakeet", "whisper"}
        assert set(BACKENDS_LOCAL.keys()) == expected

    def test_backends_docker_has_all_services(self):
        """Test BACKENDS_DOCKER contains all expected services."""
        expected = {"transcription", "vosk", "parakeet", "whisper"}
        assert set(BACKENDS_DOCKER.keys()) == expected

    def test_backend_config_has_required_fields(self):
        """Test each backend has required configuration fields."""
        required_fields = {"name", "device", "host", "port", "chunk_ms", "mode", "description"}

        for name, config in BACKENDS_LOCAL.items():
            for field in required_fields:
                assert field in config, f"Backend '{name}' missing field '{field}'"

    def test_local_backends_use_localhost(self):
        """Test local backends use localhost."""
        for name, config in BACKENDS_LOCAL.items():
            assert config["host"] == "localhost", f"Local backend '{name}' should use localhost"

    def test_docker_backends_use_service_names(self):
        """Test Docker backends use service names, not localhost."""
        # transcription gateway also uses service name
        assert BACKENDS_DOCKER["transcription"]["host"] == "transcription"
        assert BACKENDS_DOCKER["vosk"]["host"] == "vosk-asr"
        assert BACKENDS_DOCKER["parakeet"]["host"] == "parakeet-asr"
        assert BACKENDS_DOCKER["whisper"]["host"] == "whisper-asr"


class TestGetBackendConfig:
    """Tests for get_backend_config function."""

    def test_get_config_for_vosk(self):
        """Test getting vosk configuration."""
        config = get_backend_config("vosk")
        assert config["device"] == "CPU"
        assert config["mode"] == "streaming"

    def test_get_config_for_parakeet(self):
        """Test getting parakeet configuration."""
        config = get_backend_config("parakeet")
        assert config["device"] == "GPU"

    def test_get_config_for_whisper(self):
        """Test getting whisper configuration."""
        config = get_backend_config("whisper")
        assert config["device"] == "GPU"

    def test_get_config_for_transcription_gateway(self):
        """Test getting transcription gateway configuration."""
        config = get_backend_config("transcription")
        assert config["device"] == "Gateway"
        assert config["mode"] == "unified"

    def test_get_config_unknown_backend_raises(self):
        """Test unknown backend raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            get_backend_config("unknown_backend")
        assert "Unknown backend" in str(exc_info.value)


class TestFormatModelName:
    """Tests for format_model_name function."""

    def test_format_parakeet_rnnt(self):
        """Test formatting parakeet-rnnt model name."""
        result = format_model_name("nvidia/parakeet-rnnt-1.1b")
        assert result == "Parakeet RNNT 1.1B"

    def test_format_parakeet_tdt(self):
        """Test formatting parakeet-tdt model name."""
        result = format_model_name("nvidia/parakeet-tdt-1.1b")
        assert result == "Parakeet TDT 1.1B"

    def test_format_whisper_turbo(self):
        """Test formatting whisper-large-v3-turbo model name."""
        result = format_model_name("openai/whisper-large-v3-turbo")
        assert result == "Whisper Large V3 Turbo"

    def test_format_none_model_id_parakeet(self):
        """Test None model_id returns 'Parakeet' for parakeet backend."""
        result = format_model_name(None, backend="parakeet")
        assert result == "Parakeet"

    def test_format_none_model_id_whisper(self):
        """Test None model_id returns 'Whisper' for whisper backend."""
        result = format_model_name(None, backend="whisper")
        assert result == "Whisper"

    def test_format_none_model_id_no_backend(self):
        """Test None model_id with no backend defaults to Parakeet."""
        result = format_model_name(None)
        assert result == "Parakeet"

    def test_format_model_name_without_org(self):
        """Test model name without organization prefix."""
        result = format_model_name("parakeet-ctc-1.1b")
        assert result == "Parakeet CTC 1.1B"


class TestListBackends:
    """Tests for list_backends function."""

    def test_list_backends_returns_all(self):
        """Test list_backends returns all backends."""
        result = list_backends()
        expected = {"transcription", "vosk", "parakeet", "whisper"}
        assert set(result.keys()) == expected

    def test_list_backends_returns_descriptions(self):
        """Test list_backends returns descriptions."""
        result = list_backends()
        for _backend, description in result.items():
            assert isinstance(description, str)
            assert len(description) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
