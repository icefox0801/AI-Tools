"""Audio device enumeration and listing."""

import logging

logger = logging.getLogger(__name__)


def list_devices():
    """List all available audio devices for microphone and system audio capture."""
    print("\n" + "=" * 65)
    print("MICROPHONE DEVICES (for --device N)")
    print("=" * 65)

    _list_microphone_devices()

    print("\n" + "=" * 65)
    print("SYSTEM AUDIO DEVICES (for --system-audio --loopback-device N)")
    print("=" * 65)

    _list_loopback_devices()


def _list_microphone_devices():
    """List available microphone input devices."""
    try:
        import pyaudio

        p = pyaudio.PyAudio()

        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                name = info["name"]
                rate = int(info["defaultSampleRate"])
                is_loopback = "立体声混音" in name or "stereo mix" in name.lower()
                marker = " ★ STEREO MIX" if is_loopback else ""
                print(f"  [{i:2d}] {name} ({rate}Hz){marker}")

        p.terminate()
    except Exception as e:
        print(f"  Error listing microphones: {e}")


def _list_loopback_devices():
    """List available WASAPI loopback devices for system audio capture."""
    try:
        import pyaudiowpatch as pyaudio

        p = pyaudio.PyAudio()

        # Find default speaker
        default_name = ""
        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_idx = wasapi_info["defaultOutputDevice"]
            default_dev = p.get_device_info_by_index(default_idx)
            default_name = default_dev["name"]
        except Exception:
            pass

        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice"):
                name = dev["name"].replace(" [Loopback]", "")
                rate = int(dev["defaultSampleRate"])
                is_default = default_name in dev["name"]
                marker = " ★ DEFAULT" if is_default else ""
                print(f"  [{i:2d}] {name} ({rate}Hz){marker}")

        p.terminate()

        print("\n" + "=" * 65)
        print("💡 TIPS:")
        print("   • Use --system-audio to capture what you hear (no mic needed)")
        print("   • System audio gives perfect quality - no acoustic degradation")
        print("=" * 65)

    except ImportError:
        print("  (pyaudiowpatch not installed)")
        print("\n💡 Install for system audio capture:")
        print("   pip install pyaudiowpatch")
    except Exception as e:
        print(f"  Error listing loopback devices: {e}")


def get_default_microphone_info() -> dict | None:
    """Get default microphone device info."""
    try:
        import pyaudio

        p = pyaudio.PyAudio()
        info = p.get_default_input_device_info()
        p.terminate()
        return dict(info)
    except Exception:
        return None


def get_default_loopback_info() -> dict | None:
    """Get default system audio loopback device info."""
    try:
        import pyaudiowpatch as pyaudio

        p = pyaudio.PyAudio()

        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_output = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        default_name = default_output["name"]

        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice") and default_name in dev["name"]:
                p.terminate()
                return dict(dev)

        p.terminate()
        return None
    except Exception:
        return None
