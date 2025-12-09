"""
E2E Test Fixtures

Provides fixtures for spinning up Docker services and running end-to-end tests.
"""

import asyncio
import os
import subprocess
import time
from collections.abc import Generator
from typing import Any

import pytest

# Service ports (from docker-compose.yaml)
SERVICES = {
    "vosk-asr": {"port": 8001, "health": "/health", "ws": "/stream"},
    "parakeet-asr": {"port": 8002, "health": "/health", "ws": "/stream"},
    "whisper-asr": {"port": 8003, "health": "/health", "ws": "/stream"},
    "text-refiner": {"port": 8010, "health": "/health"},
}

# Default timeout for service startup
STARTUP_TIMEOUT = 120  # seconds (models take time to load)


def is_docker_running() -> bool:
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def wait_for_healthy(service: str, timeout: int = STARTUP_TIMEOUT) -> bool:
    """Wait for a service to become healthy."""
    import httpx

    port = SERVICES[service]["port"]
    health_path = SERVICES[service]["health"]
    url = f"http://localhost:{port}{health_path}"

    start = time.time()
    while time.time() - start < timeout:
        try:
            response = httpx.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)

    return False


@pytest.fixture(scope="session")
def docker_compose_file() -> str:
    """Path to docker-compose.yaml."""
    return os.path.join(os.path.dirname(__file__), "..", "..", "docker-compose.yaml")


@pytest.fixture(scope="session")
def project_root() -> str:
    """Project root directory."""
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(scope="session")
def ensure_docker() -> None:
    """Ensure Docker is running before tests."""
    if not is_docker_running():
        pytest.skip("Docker is not running")


def is_service_healthy(service: str) -> bool:
    """Check if a service is already healthy."""
    import httpx

    port = SERVICES[service]["port"]
    health_path = SERVICES[service]["health"]
    url = f"http://localhost:{port}{health_path}"
    try:
        response = httpx.get(url, timeout=2)
        return response.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def vosk_service(ensure_docker: None, project_root: str) -> Generator[dict[str, Any], None, None]:
    """
    Start Vosk ASR service for E2E tests.

    If service is already running, use it without restarting.
    Yields service config dict with host, port, endpoints.
    """
    service_name = "vosk-asr"

    # Check if already running
    if not is_service_healthy(service_name):
        # Start the service
        subprocess.run(
            ["docker", "compose", "up", "-d", service_name],
            cwd=project_root,
            capture_output=True,
        )

        # Wait for healthy
        if not wait_for_healthy(service_name):
            pytest.fail(f"{service_name} did not become healthy within {STARTUP_TIMEOUT}s")

    yield {
        "name": service_name,
        "host": "localhost",
        "port": SERVICES[service_name]["port"],
        "ws_endpoint": SERVICES[service_name]["ws"],
        "health_endpoint": SERVICES[service_name]["health"],
    }

    # Cleanup: leave running for faster re-runs


@pytest.fixture(scope="module")
def parakeet_service(
    ensure_docker: None, project_root: str
) -> Generator[dict[str, Any], None, None]:
    """Start Parakeet ASR service for E2E tests."""
    service_name = "parakeet-asr"

    if not is_service_healthy(service_name):
        subprocess.run(
            ["docker", "compose", "up", "-d", service_name],
            cwd=project_root,
            capture_output=True,
        )

        if not wait_for_healthy(service_name):
            pytest.fail(f"{service_name} did not become healthy within {STARTUP_TIMEOUT}s")

    yield {
        "name": service_name,
        "host": "localhost",
        "port": SERVICES[service_name]["port"],
        "ws_endpoint": SERVICES[service_name]["ws"],
        "health_endpoint": SERVICES[service_name]["health"],
    }


@pytest.fixture(scope="module")
def whisper_service(
    ensure_docker: None, project_root: str
) -> Generator[dict[str, Any], None, None]:
    """Start Whisper ASR service for E2E tests."""
    service_name = "whisper-asr"

    if not is_service_healthy(service_name):
        subprocess.run(
            ["docker", "compose", "up", "-d", service_name],
            cwd=project_root,
            capture_output=True,
        )

        if not wait_for_healthy(service_name):
            pytest.fail(f"{service_name} did not become healthy within {STARTUP_TIMEOUT}s")

    yield {
        "name": service_name,
        "host": "localhost",
        "port": SERVICES[service_name]["port"],
        "ws_endpoint": SERVICES[service_name]["ws"],
        "health_endpoint": SERVICES[service_name]["health"],
    }


@pytest.fixture(scope="module")
def text_refiner_service(
    ensure_docker: None, project_root: str
) -> Generator[dict[str, Any], None, None]:
    """Start Text Refiner service for E2E tests."""
    service_name = "text-refiner"

    if not is_service_healthy(service_name):
        subprocess.run(
            ["docker", "compose", "up", "-d", service_name],
            cwd=project_root,
            capture_output=True,
        )

        if not wait_for_healthy(service_name, timeout=120):
            pytest.fail(f"{service_name} did not become healthy within 120s")

    yield {
        "name": service_name,
        "host": "localhost",
        "port": SERVICES[service_name]["port"],
        "health_endpoint": SERVICES[service_name]["health"],
    }


# ============================================================================
# GPU Status Helpers
# ============================================================================


def get_gpu_status(service_config: dict[str, Any]) -> dict[str, Any]:
    """Get GPU status from a service's health endpoint.

    Returns dict with:
        - device: "cuda" or "cpu"
        - cuda_device: GPU name (if CUDA)
        - memory_gb: GPU memory allocated (if CUDA)
        - streaming_loaded: bool
        - offline_loaded: bool
    """
    import httpx

    url = f"http://{service_config['host']}:{service_config['port']}{service_config['health_endpoint']}"
    try:
        response = httpx.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {}


def assert_gpu_available(service_config: dict[str, Any]) -> None:
    """Assert that the service has CUDA GPU available.

    Raises AssertionError if GPU is not available.
    """
    status = get_gpu_status(service_config)
    assert status.get("device") == "cuda", f"Expected CUDA device, got: {status.get('device')}"
    assert "cuda_device" in status, f"Expected cuda_device in status: {status}"


def assert_model_loaded(service_config: dict[str, Any], mode: str) -> None:
    """Assert that a specific model is loaded.

    Args:
        service_config: Service configuration dict
        mode: "streaming" or "offline"
    """
    status = get_gpu_status(service_config)
    if mode == "streaming":
        assert status.get("streaming_loaded") is True, f"Expected streaming model loaded: {status}"
    elif mode == "offline":
        assert status.get("offline_loaded") is True, f"Expected offline model loaded: {status}"
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_gpu_memory_gb(service_config: dict[str, Any]) -> float:
    """Get GPU memory allocated by a service in GB."""
    status = get_gpu_status(service_config)
    return status.get("memory_gb", 0.0)


# ============================================================================
# Test Audio Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def test_audio_dir(project_root: str) -> str:
    """Path to test audio files (integration/fixtures/audio)."""
    return os.path.join(project_root, "integration", "fixtures", "audio")


@pytest.fixture(scope="session")
def hello_audio(test_audio_dir: str) -> bytes:
    """Load 'hello' test audio (16kHz mono WAV)."""
    audio_path = os.path.join(test_audio_dir, "hello_16k.wav")
    if not os.path.exists(audio_path):
        pytest.skip(f"Test audio not found: {audio_path}")
    with open(audio_path, "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def numbers_audio(test_audio_dir: str) -> bytes:
    """Load numbers test audio (16kHz mono WAV)."""
    audio_path = os.path.join(test_audio_dir, "numbers_16k.wav")
    if not os.path.exists(audio_path):
        pytest.skip(f"Test audio not found: {audio_path}")
    with open(audio_path, "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def long_audio(test_audio_dir: str) -> bytes:
    """Load long speech audio for offline transcription tests (~7 min, 16kHz mono WAV)."""
    audio_path = os.path.join(test_audio_dir, "long_speech_16k.wav")
    if not os.path.exists(audio_path):
        pytest.skip(f"Test audio not found: {audio_path}")
    with open(audio_path, "rb") as f:
        return f.read()


# ============================================================================
# Event Loop Fixture
# ============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
