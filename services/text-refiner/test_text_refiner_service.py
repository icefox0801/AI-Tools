"""
Unit tests for Text Refiner Service

Tests the processing endpoints and health check.
Uses mocked models to avoid requiring actual model files.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# ==============================================================================
# Mock External Dependencies Before Imports
# ==============================================================================

# Create mock modules for dependencies not installed locally
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.cuda.get_device_properties.return_value = MagicMock(total_memory=8 * 1024**3)
sys.modules["torch"] = mock_torch

mock_uvicorn = MagicMock()
sys.modules["uvicorn"] = mock_uvicorn

mock_punctuators = MagicMock()
sys.modules["punctuators"] = mock_punctuators

mock_transformers = MagicMock()
sys.modules["transformers"] = mock_transformers

from fastapi.testclient import TestClient

# ==============================================================================
# Mock Models
# ==============================================================================


class MockPunctuationModel:
    """Mock punctuators ONNX model."""

    def infer(self, texts: list[str]) -> list[list[str]]:
        """Return punctuated text."""
        results = []
        for text in texts:
            # Simple mock: capitalize first letter, add period
            if text.strip():
                punctuated = text.strip().capitalize()
                if not punctuated.endswith((".", "!", "?")):
                    punctuated += "."
                results.append([punctuated])
            else:
                results.append([""])
        return results


class MockCorrectionModel:
    """Mock T5 correction model."""

    def __init__(self):
        self._device = "cpu"

    def parameters(self):
        """Return mock parameter with device."""
        mock_param = MagicMock()
        mock_param.device = self._device
        return iter([mock_param])

    def to(self, device):
        self._device = device
        return self

    def half(self):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        """Return mock output tensor."""
        return [[1, 2, 3]]  # Mock token IDs


class MockTokenizer:
    """Mock HuggingFace tokenizer."""

    def __call__(self, text, **kwargs):
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

    def decode(self, tokens, **kwargs):
        # Simple mock: return original text (no actual correction)
        return "Hello world corrected."


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_models():
    """Patch model loading functions."""
    mock_punct = MockPunctuationModel()
    mock_correction = MockCorrectionModel()
    mock_tokenizer = MockTokenizer()

    with patch.dict("sys.modules", {"torch": MagicMock()}):
        with patch("text_refiner_service.get_punctuation_model", return_value=mock_punct):
            with patch(
                "text_refiner_service.get_correction_model",
                return_value=(mock_correction, mock_tokenizer),
            ):
                yield


@pytest.fixture
def app():
    """Create FastAPI app with mocked models."""
    # Remove cached module if exists
    if "text_refiner_service" in sys.modules:
        del sys.modules["text_refiner_service"]

    # Mock the model loading
    with patch.dict(
        "os.environ",
        {
            "PUNCTUATION_MODEL": "pcs_en",
            "ENABLE_CORRECTION": "false",  # Disable for simpler tests
        },
    ):
        import text_refiner_service

        # Replace model getters with mocks
        text_refiner_service._punctuation_model = MockPunctuationModel()

        yield text_refiner_service.app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


# ==============================================================================
# Health Endpoint Tests
# ==============================================================================


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self, client):
        """Health check returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"

    def test_health_returns_device(self, client):
        """Health check includes device information."""
        response = client.get("/health")
        data = response.json()

        assert "device" in data

    def test_health_returns_correction_enabled(self, client):
        """Health check includes correction_enabled flag."""
        response = client.get("/health")
        data = response.json()

        assert "correction_enabled" in data


# ==============================================================================
# Info Endpoint Tests
# ==============================================================================


class TestInfoEndpoint:
    """Tests for the /info endpoint."""

    def test_info_returns_service_name(self, client):
        """Info returns service name."""
        response = client.get("/info")
        assert response.status_code == 200

        data = response.json()
        assert data["service"] == "text-refiner"

    def test_info_returns_api_version(self, client):
        """Info returns API version."""
        response = client.get("/info")
        data = response.json()

        assert "api_version" in data
        assert data["api_version"] == "1.0"

    def test_info_returns_models(self, client):
        """Info returns model information."""
        response = client.get("/info")
        data = response.json()

        assert "models" in data
        assert "punctuation" in data["models"]
        assert "correction" in data["models"]


# ==============================================================================
# Process Endpoint Tests
# ==============================================================================


class TestProcessEndpoint:
    """Tests for the /process endpoint."""

    def test_process_returns_text(self, client):
        """Process endpoint returns processed text."""
        response = client.post(
            "/process", json={"text": "hello world", "punctuate": True, "correct": False}
        )
        assert response.status_code == 200

        data = response.json()
        assert "text" in data
        assert "original" in data
        assert data["original"] == "hello world"

    def test_process_returns_latency(self, client):
        """Process endpoint returns latency."""
        response = client.post("/process", json={"text": "hello", "correct": False})
        data = response.json()

        assert "latency_ms" in data
        assert isinstance(data["latency_ms"], float)

    def test_process_punctuates_text(self, client):
        """Process endpoint punctuates text."""
        response = client.post(
            "/process", json={"text": "hello world", "punctuate": True, "correct": False}
        )
        data = response.json()

        # Mock adds period and capitalizes
        assert data["punctuated"] is True

    def test_process_empty_text(self, client):
        """Process endpoint handles empty text."""
        response = client.post("/process", json={"text": "", "correct": False})
        assert response.status_code == 200

        data = response.json()
        assert data["text"] == ""


# ==============================================================================
# Punctuate Endpoint Tests
# ==============================================================================


class TestPunctuateEndpoint:
    """Tests for the /punctuate endpoint."""

    def test_punctuate_returns_text(self, client):
        """Punctuate endpoint returns text."""
        response = client.post("/punctuate", json={"text": "hello"})
        assert response.status_code == 200

        data = response.json()
        assert "text" in data
        assert "original" in data

    def test_punctuate_returns_latency(self, client):
        """Punctuate endpoint returns latency."""
        response = client.post("/punctuate", json={"text": "hello"})
        data = response.json()

        assert "latency_ms" in data


# ==============================================================================
# Batch Endpoint Tests
# ==============================================================================


class TestBatchEndpoint:
    """Tests for the /batch endpoint."""

    def test_batch_processes_multiple_texts(self, client):
        """Batch endpoint processes multiple texts."""
        response = client.post(
            "/batch",
            json={"texts": ["hello", "world", "test"], "punctuate": True, "correct": False},
        )
        assert response.status_code == 200

        data = response.json()
        assert "texts" in data
        assert len(data["texts"]) == 3

    def test_batch_returns_latency(self, client):
        """Batch endpoint returns latency."""
        response = client.post("/batch", json={"texts": ["hello"], "correct": False})
        data = response.json()

        assert "latency_ms" in data


# ==============================================================================
# Request/Response Model Tests
# ==============================================================================


class TestRequestModels:
    """Tests for request/response model validation."""

    def test_process_request_defaults(self, client):
        """Process request has correct defaults."""
        response = client.post("/process", json={"text": "test"})
        # Should work with minimal input
        assert response.status_code == 200

    def test_process_request_with_context(self, client):
        """Process request accepts context."""
        response = client.post(
            "/process",
            json={"text": "test", "context": "previous sentence", "correct": False},
        )
        assert response.status_code == 200


# ==============================================================================
# Error Handling Tests
# ==============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_text_field(self, client):
        """Missing text field returns 422."""
        response = client.post("/process", json={})
        assert response.status_code == 422

    def test_invalid_json(self, client):
        """Invalid JSON returns 422."""
        response = client.post(
            "/process", content="not json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


# ==============================================================================
# Run Standalone
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
