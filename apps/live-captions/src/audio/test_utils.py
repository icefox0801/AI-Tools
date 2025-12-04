"""Unit tests for audio utility functions."""

import numpy as np

from .utils import (
    CHUNK_DURATION_MS,
    TARGET_SAMPLE_RATE,
    calculate_chunk_size,
    resample_audio,
    stereo_to_mono,
)


class TestResampleAudio:
    """Tests for resample_audio function."""

    def test_same_rate_returns_unchanged(self):
        """Test that same sample rate returns unchanged data."""
        # Create 16-bit PCM audio data (1 second at 16kHz)
        samples = np.array([0, 1000, -1000, 500, -500], dtype=np.int16)
        audio_bytes = samples.tobytes()

        result = resample_audio(audio_bytes, 16000, 16000)

        assert result == audio_bytes

    def test_downsample_48k_to_16k(self):
        """Test downsampling from 48kHz to 16kHz."""
        # Create a simple sine wave at 48kHz
        duration_samples = 480  # 10ms at 48kHz
        samples = np.sin(np.linspace(0, 2 * np.pi, duration_samples)) * 16000
        audio_bytes = samples.astype(np.int16).tobytes()

        result = resample_audio(audio_bytes, 48000, 16000)

        # Result should be roughly 1/3 the length
        result_samples = np.frombuffer(result, dtype=np.int16)
        expected_length = duration_samples // 3
        assert abs(len(result_samples) - expected_length) <= 1

    def test_upsample_8k_to_16k(self):
        """Test upsampling from 8kHz to 16kHz."""
        # Create audio at 8kHz
        duration_samples = 80  # 10ms at 8kHz
        samples = np.linspace(-10000, 10000, duration_samples).astype(np.int16)
        audio_bytes = samples.tobytes()

        result = resample_audio(audio_bytes, 8000, 16000)

        # Result should be roughly 2x the length
        result_samples = np.frombuffer(result, dtype=np.int16)
        expected_length = duration_samples * 2
        assert abs(len(result_samples) - expected_length) <= 1

    def test_preserves_dtype(self):
        """Test that output is 16-bit PCM."""
        samples = np.array([0, 1000, -1000], dtype=np.int16)
        audio_bytes = samples.tobytes()

        result = resample_audio(audio_bytes, 8000, 16000)

        # Should be valid int16 data
        result_samples = np.frombuffer(result, dtype=np.int16)
        assert result_samples.dtype == np.int16


class TestStereoToMono:
    """Tests for stereo_to_mono function."""

    def test_converts_stereo_to_mono(self):
        """Test basic stereo to mono conversion."""
        # Create stereo audio: L=1000, R=2000 for each sample
        left = np.array([1000, 2000, 3000], dtype=np.int16)
        right = np.array([2000, 4000, 6000], dtype=np.int16)
        stereo = np.column_stack((left, right)).flatten().tobytes()

        result = stereo_to_mono(stereo)

        # Average of L and R channels
        expected = np.array([1500, 3000, 4500], dtype=np.int16)
        result_samples = np.frombuffer(result, dtype=np.int16)
        np.testing.assert_array_equal(result_samples, expected)

    def test_identical_channels_unchanged(self):
        """Test that identical L/R channels produce same output."""
        samples = np.array([1000, 1000, 2000, 2000, 3000, 3000], dtype=np.int16)
        stereo = samples.tobytes()

        result = stereo_to_mono(stereo)

        expected = np.array([1000, 2000, 3000], dtype=np.int16)
        result_samples = np.frombuffer(result, dtype=np.int16)
        np.testing.assert_array_equal(result_samples, expected)

    def test_silence_produces_silence(self):
        """Test that silent stereo produces silent mono."""
        stereo = np.zeros(10, dtype=np.int16).tobytes()

        result = stereo_to_mono(stereo)

        result_samples = np.frombuffer(result, dtype=np.int16)
        assert np.all(result_samples == 0)

    def test_output_is_half_length(self):
        """Test that mono output is half the stereo length."""
        stereo_samples = 100
        stereo = np.zeros(stereo_samples, dtype=np.int16).tobytes()

        result = stereo_to_mono(stereo)

        result_samples = np.frombuffer(result, dtype=np.int16)
        assert len(result_samples) == stereo_samples // 2


class TestCalculateChunkSize:
    """Tests for calculate_chunk_size function."""

    def test_default_duration(self):
        """Test chunk size with default duration."""
        chunk_size = calculate_chunk_size(16000)

        # Default is 100ms, so 16000 * 0.1 = 1600
        expected = int(16000 * CHUNK_DURATION_MS / 1000)
        assert chunk_size == expected

    def test_custom_duration(self):
        """Test chunk size with custom duration."""
        chunk_size = calculate_chunk_size(16000, duration_ms=200)

        # 200ms at 16kHz = 3200 samples
        assert chunk_size == 3200

    def test_different_sample_rates(self):
        """Test chunk sizes for different sample rates."""
        # 48kHz at 100ms
        chunk_48k = calculate_chunk_size(48000)
        expected_48k = int(48000 * CHUNK_DURATION_MS / 1000)
        assert chunk_48k == expected_48k

        # 8kHz at 100ms
        chunk_8k = calculate_chunk_size(8000)
        expected_8k = int(8000 * CHUNK_DURATION_MS / 1000)
        assert chunk_8k == expected_8k


class TestConstants:
    """Tests for module constants."""

    def test_target_sample_rate(self):
        """Test target sample rate constant."""
        assert TARGET_SAMPLE_RATE == 16000

    def test_chunk_duration(self):
        """Test chunk duration constant."""
        assert CHUNK_DURATION_MS == 100
