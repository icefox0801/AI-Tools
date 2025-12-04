"""Unit tests for audio-notes UI main module utility functions."""

from pathlib import Path
from unittest.mock import patch

import pytest


class TestFormatRecordingTitle:
    """Tests for format_recording_title function."""

    def test_recording_pattern(self):
        """Test formatting of recording_YYYYMMDD_HHMMSS pattern."""
        from ui.main import format_recording_title
        
        result = format_recording_title("recording_20251203_205252.wav")
        assert "Recording" in result
        assert "Dec" in result or "2025" in result

    def test_uploaded_pattern(self):
        """Test formatting of uploaded_YYYYMMDD_HHMMSS pattern."""
        from ui.main import format_recording_title
        
        result = format_recording_title("uploaded_20251201_143000.mp3")
        assert "Uploaded" in result
        assert "Dec" in result or "2025" in result

    def test_fallback_for_unknown_pattern(self):
        """Test fallback formatting for unknown pattern."""
        from ui.main import format_recording_title
        
        result = format_recording_title("my_audio.wav")
        # Should clean up underscores and title case
        assert "My Audio" in result or "my_audio" in result.lower()

    def test_no_extension_in_result(self):
        """Test that extension is stripped from stem."""
        from ui.main import format_recording_title
        
        result = format_recording_title("test_file.wav")
        assert ".wav" not in result

    def test_case_insensitive_pattern(self):
        """Test pattern matching is case insensitive."""
        from ui.main import format_recording_title
        
        result = format_recording_title("RECORDING_20251203_120000.wav")
        assert "Recording" in result


class TestBatchTranscribeStreaming:
    """Tests for batch_transcribe_streaming generator function."""

    def test_empty_selection_yields_warning(self):
        """Test that empty selection yields warning message."""
        from ui.main import batch_transcribe_streaming
        
        results = list(batch_transcribe_streaming([]))
        assert len(results) == 1
        assert "No files selected" in results[0][0]

    def test_yields_progress_updates(self):
        """Test that function yields progress updates during transcription."""
        from ui.main import batch_transcribe_streaming
        
        with (
            patch("ui.main.unload_asr_model", return_value=(True, "Unloaded")),
            patch("ui.main.check_parakeet_health", return_value=(True, "OK")),
            patch("ui.main.transcribe_audio", return_value=("Test transcript", 5.0)),
            patch("pathlib.Path.write_text"),
        ):
            results = list(batch_transcribe_streaming(["/fake/path/test.wav"], "parakeet"))
            
            # Should have multiple yields: GPU prep, transcribing progress, final result
            assert len(results) >= 2
            # First should be GPU preparation
            assert "GPU" in results[0][0] or "Preparing" in results[0][0]

    def test_handles_transcription_error(self):
        """Test that transcription errors are captured."""
        from ui.main import batch_transcribe_streaming
        
        with (
            patch("ui.main.unload_asr_model", return_value=(True, "Unloaded")),
            patch("ui.main.check_parakeet_health", return_value=(True, "OK")),
            patch("ui.main.transcribe_audio", side_effect=Exception("Test error")),
        ):
            results = list(batch_transcribe_streaming(["/fake/path/test.wav"], "parakeet"))
            
            # Should contain error marker
            final_status = results[-1][0]
            assert "❌" in final_status or "error" in final_status.lower()

    def test_service_unavailable(self):
        """Test handling when ASR service is unavailable."""
        from ui.main import batch_transcribe_streaming
        
        with (
            patch("ui.main.unload_asr_model", return_value=(True, "Unloaded")),
            patch("ui.main.check_parakeet_health", return_value=(False, "Service unavailable")),
        ):
            results = list(batch_transcribe_streaming(["/fake/path/test.wav"], "parakeet"))
            
            # Should yield error about service
            assert any("❌" in r[0] for r in results)


class TestGenerateSummary:
    """Tests for generate_summary generator function."""

    def test_empty_transcript_yields_warning(self):
        """Test that empty transcript yields warning."""
        from ui.main import generate_summary
        
        results = list(generate_summary(""))
        assert len(results) == 1
        assert "No transcript" in results[0]

    def test_ollama_unavailable(self):
        """Test handling when Ollama is unavailable."""
        from ui.main import generate_summary
        
        with patch("ui.main.check_ollama_health", return_value=(False, "Ollama not running")):
            results = list(generate_summary("Test transcript"))
            
            assert len(results) == 1
            assert "❌" in results[0]

    def test_calls_summarize_streaming(self):
        """Test that function calls summarize_streaming correctly."""
        from ui.main import generate_summary
        
        def mock_summarize(*args, **kwargs):
            yield "Summary chunk 1"
            yield "Summary chunk 2"
        
        with (
            patch("ui.main.check_ollama_health", return_value=(True, "OK")),
            patch("ui.main.summarize_streaming", side_effect=mock_summarize),
        ):
            results = list(generate_summary("Test transcript", "Custom prompt", "test-model"))
            
            assert "Summary chunk 1" in results
            assert "Summary chunk 2" in results
