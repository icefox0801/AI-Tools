"""Unit tests for audio-notes API routes."""

import io
import os
import sys
import tempfile
import wave
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.routes import setup_api_routes


@pytest.fixture
def test_app():
    """Create a test FastAPI app with routes configured."""
    app = FastAPI()
    setup_api_routes(app)
    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


@pytest.fixture
def temp_recordings_dir():
    """Create a temporary recordings directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def create_wav_bytes(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Create valid WAV file bytes for testing."""
    buffer = io.BytesIO()
    num_samples = int(sample_rate * duration_seconds)
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * num_samples)
    buffer.seek(0)
    return buffer.read()


class TestHealthEndpoint:
    """Tests for /api/health endpoint."""

    def test_health_returns_ok(self, client, temp_recordings_dir):
        """Test health check returns success."""
        with patch("api.routes.RECORDINGS_DIR", temp_recordings_dir):
            response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "recordings_dir" in data
        assert "recordings_count" in data

    def test_health_counts_wav_files(self, client, temp_recordings_dir):
        """Test health check counts WAV files."""
        # Create some WAV files
        (temp_recordings_dir / "test1.wav").write_bytes(b"fake")
        (temp_recordings_dir / "test2.wav").write_bytes(b"fake")
        
        with patch("api.routes.RECORDINGS_DIR", temp_recordings_dir):
            response = client.get("/api/health")
        
        assert response.status_code == 200
        assert response.json()["recordings_count"] == 2


class TestRecordingsEndpoint:
    """Tests for /api/recordings endpoint."""

    def test_list_recordings_empty(self, client, temp_recordings_dir):
        """Test listing recordings when empty."""
        with (
            patch("api.routes.RECORDINGS_DIR", temp_recordings_dir),
            patch("api.routes.list_recordings", return_value=[]),
        ):
            response = client.get("/api/recordings")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["recordings"] == []

    def test_list_recordings_with_files(self, client, temp_recordings_dir):
        """Test listing recordings with files."""
        mock_recordings = [
            {"filename": "test1.wav", "size_mb": 1.5, "duration": 60.0},
            {"filename": "test2.wav", "size_mb": 2.0, "duration": 120.0},
        ]
        
        with (
            patch("api.routes.RECORDINGS_DIR", temp_recordings_dir),
            patch("api.routes.list_recordings", return_value=mock_recordings),
        ):
            response = client.get("/api/recordings")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["recordings"]) == 2

    def test_list_recordings_error(self, client, temp_recordings_dir):
        """Test listing recordings when error occurs."""
        with (
            patch("api.routes.RECORDINGS_DIR", temp_recordings_dir),
            patch("api.routes.list_recordings", side_effect=Exception("Test error")),
        ):
            response = client.get("/api/recordings")
        
        assert response.status_code == 500


class TestUploadAudioEndpoint:
    """Tests for /api/upload-audio endpoint."""

    def test_upload_audio_success(self, client, temp_recordings_dir):
        """Test successful audio upload."""
        wav_bytes = create_wav_bytes(1.0)
        
        with (
            patch("api.routes.RECORDINGS_DIR", temp_recordings_dir),
            patch("api.routes.get_audio_duration", return_value=1.0),
        ):
            response = client.post(
                "/api/upload-audio",
                files={"audio": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
                data={"filename": "test.wav", "append": "false"},
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["filename"] == "test.wav"
        assert data["duration"] == 1.0

    def test_upload_audio_adds_wav_extension(self, client, temp_recordings_dir):
        """Test that .wav extension is added if missing."""
        wav_bytes = create_wav_bytes(1.0)
        
        with (
            patch("api.routes.RECORDINGS_DIR", temp_recordings_dir),
            patch("api.routes.get_audio_duration", return_value=1.0),
        ):
            response = client.post(
                "/api/upload-audio",
                files={"audio": ("test", io.BytesIO(wav_bytes), "audio/wav")},
                data={"filename": "test_no_extension", "append": "false"},
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test_no_extension.wav"

    def test_upload_audio_append(self, client, temp_recordings_dir):
        """Test appending audio to existing file."""
        # Create initial WAV file
        wav_bytes = create_wav_bytes(1.0)
        existing_path = temp_recordings_dir / "existing.wav"
        existing_path.write_bytes(wav_bytes)
        
        with (
            patch("api.routes.RECORDINGS_DIR", temp_recordings_dir),
            patch("api.routes.get_audio_duration", return_value=2.0),
        ):
            response = client.post(
                "/api/upload-audio",
                files={"audio": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
                data={"filename": "existing.wav", "append": "true"},
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_upload_audio_creates_directory(self, client, temp_recordings_dir):
        """Test that recordings directory is created if it doesn't exist."""
        new_dir = temp_recordings_dir / "new_subdir"
        wav_bytes = create_wav_bytes(1.0)
        
        with (
            patch("api.routes.RECORDINGS_DIR", new_dir),
            patch("api.routes.get_audio_duration", return_value=1.0),
        ):
            response = client.post(
                "/api/upload-audio",
                files={"audio": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
                data={"filename": "test.wav", "append": "false"},
            )
        
        assert response.status_code == 200
        assert new_dir.exists()


class TestStaticFiles:
    """Tests for static file serving endpoints."""

    def test_favicon_not_found(self, client):
        """Test favicon returns 404 when not found."""
        with patch("api.routes.APP_DIR", Path("/nonexistent")):
            response = client.get("/favicon.ico")
        
        assert response.status_code == 404

    def test_icon_192_not_found(self, client):
        """Test icon-192 returns 404 when not found."""
        with patch("api.routes.APP_DIR", Path("/nonexistent")):
            response = client.get("/icon-192.png")
        
        assert response.status_code == 404

    def test_icon_512_not_found(self, client):
        """Test icon-512 returns 404 when not found."""
        with patch("api.routes.APP_DIR", Path("/nonexistent")):
            response = client.get("/icon-512.png")
        
        assert response.status_code == 404

    def test_manifest_not_found(self, client):
        """Test manifest returns 404 when not found."""
        with patch("api.routes.APP_DIR", Path("/nonexistent")):
            response = client.get("/manifest.json")
        
        assert response.status_code == 404

    def test_favicon_found(self, client, temp_recordings_dir):
        """Test favicon is served when found."""
        favicon_path = temp_recordings_dir / "favicon.ico"
        favicon_path.write_bytes(b"\x00\x00\x01\x00")  # Minimal ICO header
        
        with patch("api.routes.APP_DIR", temp_recordings_dir):
            response = client.get("/favicon.ico")
        
        assert response.status_code == 200

    def test_manifest_found(self, client, temp_recordings_dir):
        """Test manifest is served when found."""
        manifest_path = temp_recordings_dir / "manifest.json"
        manifest_path.write_text('{"name": "Test App"}')
        
        with patch("api.routes.APP_DIR", temp_recordings_dir):
            response = client.get("/manifest.json")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/manifest+json"
