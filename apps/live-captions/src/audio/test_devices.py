"""
Unit tests for audio devices module.

Tests the device enumeration functions including:
- Microphone device listing
- Loopback device listing
- Default device info retrieval
"""

from unittest.mock import MagicMock, patch

import pytest


class TestListDevices:
    """Tests for list_devices function."""

    def test_calls_both_listers(self, capsys):
        """Test that list_devices calls both microphone and loopback listers."""
        with (
            patch("src.audio.devices._list_microphone_devices") as mock_mic,
            patch("src.audio.devices._list_loopback_devices") as mock_loopback,
        ):
            from src.audio.devices import list_devices

            list_devices()

            mock_mic.assert_called_once()
            mock_loopback.assert_called_once()

    def test_prints_headers(self, capsys):
        """Test that list_devices prints section headers."""
        with (
            patch("src.audio.devices._list_microphone_devices"),
            patch("src.audio.devices._list_loopback_devices"),
        ):
            from src.audio.devices import list_devices

            list_devices()
            captured = capsys.readouterr()

            assert "MICROPHONE DEVICES" in captured.out
            assert "SYSTEM AUDIO DEVICES" in captured.out


class TestListMicrophoneDevices:
    """Tests for _list_microphone_devices function."""

    def test_lists_input_devices(self, capsys):
        """Test listing microphone input devices."""
        mock_pyaudio = MagicMock()
        mock_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_instance
        mock_instance.get_device_count.return_value = 2
        mock_instance.get_device_info_by_index.side_effect = [
            {"name": "Microphone", "maxInputChannels": 1, "defaultSampleRate": 48000},
            {"name": "Speakers", "maxInputChannels": 0, "defaultSampleRate": 48000},
        ]

        with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
            from src.audio.devices import _list_microphone_devices

            _list_microphone_devices()
            captured = capsys.readouterr()

            assert "Microphone" in captured.out
            assert "Speakers" not in captured.out

    def test_shows_stereo_mix_marker(self, capsys):
        """Test that stereo mix devices get marked."""
        mock_pyaudio = MagicMock()
        mock_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_instance
        mock_instance.get_device_count.return_value = 2
        mock_instance.get_device_info_by_index.side_effect = [
            {"name": "Microphone", "maxInputChannels": 1, "defaultSampleRate": 48000},
            {"name": "Stereo Mix", "maxInputChannels": 2, "defaultSampleRate": 48000},
        ]

        with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
            from src.audio.devices import _list_microphone_devices

            _list_microphone_devices()
            captured = capsys.readouterr()

            assert "STEREO MIX" in captured.out

    def test_handles_pyaudio_error(self, capsys):
        """Test handling of PyAudio errors."""
        mock_pyaudio = MagicMock()
        mock_pyaudio.PyAudio.side_effect = Exception("No audio device")

        with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
            from src.audio.devices import _list_microphone_devices

            _list_microphone_devices()
            captured = capsys.readouterr()

            assert "Error listing microphones" in captured.out

    def test_terminates_pyaudio(self):
        """Test that PyAudio is properly terminated."""
        mock_pyaudio = MagicMock()
        mock_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_instance
        mock_instance.get_device_count.return_value = 0

        with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
            from src.audio.devices import _list_microphone_devices

            _list_microphone_devices()

            mock_instance.terminate.assert_called_once()


class TestListLoopbackDevices:
    """Tests for _list_loopback_devices function."""

    def test_lists_loopback_devices(self, capsys):
        """Test listing WASAPI loopback devices."""
        mock_pyaudio = MagicMock()
        mock_pyaudio.paWASAPI = 1
        mock_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_instance
        mock_instance.get_host_api_info_by_type.return_value = {"defaultOutputDevice": 0}
        mock_instance.get_device_info_by_index.side_effect = [
            {"name": "Speakers"},  # default output
            {"name": "Speakers [Loopback]", "isLoopbackDevice": True, "defaultSampleRate": 48000},
        ]
        mock_instance.get_device_count.return_value = 2

        with patch.dict("sys.modules", {"pyaudiowpatch": mock_pyaudio}):
            from src.audio.devices import _list_loopback_devices

            _list_loopback_devices()
            captured = capsys.readouterr()

            assert "Speakers" in captured.out

    def test_shows_default_marker(self, capsys):
        """Test that default output device's loopback gets marked."""
        mock_pyaudio = MagicMock()
        mock_pyaudio.paWASAPI = 1
        mock_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_instance
        mock_instance.get_host_api_info_by_type.return_value = {"defaultOutputDevice": 0}
        mock_instance.get_device_count.return_value = 2
        mock_instance.get_device_info_by_index.side_effect = [
            {"name": "Default Speakers"},  # default output lookup
            {
                "name": "Default Speakers [Loopback]",
                "isLoopbackDevice": True,
                "defaultSampleRate": 48000,
            },
            {"name": "Other [Loopback]", "isLoopbackDevice": True, "defaultSampleRate": 48000},
        ]

        with patch.dict("sys.modules", {"pyaudiowpatch": mock_pyaudio}):
            from src.audio.devices import _list_loopback_devices

            _list_loopback_devices()
            captured = capsys.readouterr()

            assert "DEFAULT" in captured.out

    def test_shows_tips(self, capsys):
        """Test that tips are shown."""
        mock_pyaudio = MagicMock()
        mock_pyaudio.paWASAPI = 1
        mock_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_instance
        mock_instance.get_device_count.return_value = 0
        mock_instance.get_host_api_info_by_type.return_value = {"defaultOutputDevice": 0}
        mock_instance.get_device_info_by_index.return_value = {"name": "Speakers"}

        with patch.dict("sys.modules", {"pyaudiowpatch": mock_pyaudio}):
            from src.audio.devices import _list_loopback_devices

            _list_loopback_devices()
            captured = capsys.readouterr()

            assert "TIPS" in captured.out

    def test_handles_import_error(self, capsys):
        """Test handling when pyaudiowpatch is not installed."""
        import builtins
        import src.audio.devices as devices_module

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyaudiowpatch":
                raise ImportError("No module named 'pyaudiowpatch'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            devices_module._list_loopback_devices()
            captured = capsys.readouterr()

        assert "pyaudiowpatch not installed" in captured.out

    def test_terminates_pyaudio(self):
        """Test that PyAudio is properly terminated."""
        mock_pyaudio = MagicMock()
        mock_pyaudio.paWASAPI = 1
        mock_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_instance
        mock_instance.get_device_count.return_value = 0
        mock_instance.get_host_api_info_by_type.return_value = {"defaultOutputDevice": 0}
        mock_instance.get_device_info_by_index.return_value = {"name": "Speakers"}

        with patch.dict("sys.modules", {"pyaudiowpatch": mock_pyaudio}):
            from src.audio.devices import _list_loopback_devices

            _list_loopback_devices()

            mock_instance.terminate.assert_called_once()


class TestGetDefaultMicrophoneInfo:
    """Tests for get_default_microphone_info function."""

    def test_returns_device_info(self):
        """Test returning default microphone info."""
        mock_pyaudio = MagicMock()
        mock_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_instance
        mock_instance.get_default_input_device_info.return_value = {
            "name": "Default Mic",
            "index": 0,
            "maxInputChannels": 2,
        }

        with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
            from src.audio.devices import get_default_microphone_info

            result = get_default_microphone_info()

            assert result is not None
            assert result["name"] == "Default Mic"
            mock_instance.terminate.assert_called_once()

    def test_returns_none_on_error(self):
        """Test returning None when no microphone available."""
        mock_pyaudio = MagicMock()
        mock_pyaudio.PyAudio.side_effect = Exception("No audio device")

        with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
            from src.audio.devices import get_default_microphone_info

            result = get_default_microphone_info()

            assert result is None


class TestGetDefaultLoopbackInfo:
    """Tests for get_default_loopback_info function."""

    def test_returns_loopback_info(self):
        """Test returning default loopback device info."""
        mock_pyaudio = MagicMock()
        mock_pyaudio.paWASAPI = 1
        mock_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_instance
        mock_instance.get_host_api_info_by_type.return_value = {"defaultOutputDevice": 0}
        mock_instance.get_device_count.return_value = 2
        mock_instance.get_device_info_by_index.side_effect = [
            {"name": "Speakers"},  # default output
            {"name": "Regular Device", "isLoopbackDevice": False},
            {"name": "Speakers [Loopback]", "isLoopbackDevice": True},
        ]

        with patch.dict("sys.modules", {"pyaudiowpatch": mock_pyaudio}):
            from src.audio.devices import get_default_loopback_info

            result = get_default_loopback_info()

            assert result is not None
            assert result["isLoopbackDevice"] is True
            mock_instance.terminate.assert_called()

    def test_returns_none_when_no_loopback(self):
        """Test returning None when no loopback device found."""
        mock_pyaudio = MagicMock()
        mock_pyaudio.paWASAPI = 1
        mock_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_instance
        mock_instance.get_host_api_info_by_type.return_value = {"defaultOutputDevice": 0}
        mock_instance.get_device_count.return_value = 1
        mock_instance.get_device_info_by_index.side_effect = [
            {"name": "Speakers"},  # default output
            {"name": "Not Loopback", "isLoopbackDevice": False},
        ]

        with patch.dict("sys.modules", {"pyaudiowpatch": mock_pyaudio}):
            from src.audio.devices import get_default_loopback_info

            result = get_default_loopback_info()

            assert result is None

    def test_returns_none_on_error(self):
        """Test returning None when pyaudiowpatch not available."""
        mock_pyaudio = MagicMock()
        mock_pyaudio.PyAudio.side_effect = Exception("Not available")

        with patch.dict("sys.modules", {"pyaudiowpatch": mock_pyaudio}):
            from src.audio.devices import get_default_loopback_info

            result = get_default_loopback_info()

            assert result is None
