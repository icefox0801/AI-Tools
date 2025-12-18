"""Audio upload UI module for benchmark notebook."""

import io
import numpy as np
import soundfile as sf
import librosa
import requests
from ipywidgets import Layout, FileUpload, Label, GridBox
from IPython.display import display, Audio

# Global widget reference for the two-cell pattern
_upload_widget = None


def create_upload_widget():
    """
    Create and display a fresh upload widget.

    Call this in Cell 1 to show the upload UI.
    Then call get_uploaded_audio() in Cell 2 to process.

    Returns:
        FileUpload: The widget (also stored globally)
    """
    global _upload_widget

    # Always create a fresh widget
    _upload_widget = FileUpload(
        accept=".wav,.mp3,.m4a,.ogg,.flac", multiple=False, description="Select Audio"
    )

    # Combine widgets
    header = Label("📤 Audio Upload")
    upload_ui = GridBox(
        [header, _upload_widget], layout=Layout(grid_template_columns="150px 150px")
    )

    return upload_ui


def get_uploaded_audio(target_sample_rate=16000, transcribe=True):
    """
    Get the uploaded audio from the widget - blocking-style like requests.post().

    This function processes the uploaded file and returns the result immediately.
    Call this after the user has uploaded a file via create_upload_widget().

    Args:
        target_sample_rate: Target sample rate for processing
        transcribe: Whether to transcribe with Whisper

    Returns:
        tuple: (audio_data, text) ready for benchmarking

    Raises:
        ValueError: If no file has been uploaded yet
    """
    global _upload_widget

    if _upload_widget is None:
        raise ValueError("No upload widget created! Run create_upload_widget() first.")

    if not _upload_widget.value:
        raise ValueError("No file uploaded yet! Upload a file first, then run this cell.")

    # Process uploaded file
    audio_data, sample_rate, filename = process_uploaded_file(_upload_widget)

    print(f"🔄 Processing: {filename}")

    # Resample to target rate
    audio_data = resample_audio(audio_data, sample_rate, target_sample_rate)

    print(f"✅ Audio ready: {len(audio_data)/target_sample_rate:.2f}s at {target_sample_rate}Hz")

    # Transcribe if requested
    text = ""
    if transcribe:
        text = transcribe_with_whisper(audio_data, target_sample_rate)

    # Display audio player
    print(f"\n🔊 Audio Playback")
    display(Audio(data=audio_data, rate=target_sample_rate, autoplay=False))

    # Display transcript preview (1 line max)
    if text:
        preview_length = 60
        if len(text) > preview_length:
            print(f"📄 Transcript: {text[:preview_length]}... ({len(text)} chars)")
        else:
            print(f"📄 Transcript: {text}")

    print(f"✅ Ready for benchmarking!")

    return audio_data, text


def process_uploaded_file(file_upload):
    """
    Process the uploaded audio file from the widget reference.

    Args:
        file_upload: FileUpload widget with uploaded file

    Returns:
        tuple: (audio_data, sample_rate, filename)
            - audio_data: numpy array of audio samples (mono, float32)
            - sample_rate: original sample rate of the audio file
            - filename: name of the uploaded file

    Raises:
        ValueError: If no file has been uploaded
    """
    if not file_upload.value:
        raise ValueError("No file uploaded yet! Please upload a file first.")

    # Get uploaded file data - FileUpload.value is a tuple of file metadata dicts
    uploaded_files = file_upload.value
    if len(uploaded_files) == 0:
        raise ValueError("No file uploaded yet! Please upload a file first.")

    # Get first uploaded file
    file_info = uploaded_files[0]
    filename = file_info["name"]
    audio_bytes = file_info["content"]

    print(f"📁 Processing: {filename} ({len(audio_bytes)/1024:.2f} KB)")

    # Read audio with soundfile
    audio_file = io.BytesIO(audio_bytes)
    audio_data, sample_rate = sf.read(audio_file)

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        print(f"🔄 Converting from {audio_data.shape[1]} channels to mono")
        audio_data = np.mean(audio_data, axis=1)

    # Ensure float32 format
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # Normalize to [-1, 1] range if needed
    max_val = np.abs(audio_data).max()
    if max_val > 1.0:
        audio_data = audio_data / max_val

    print(f"✅ Loaded: {len(audio_data)/sample_rate:.2f}s at {sample_rate}Hz")

    return audio_data, sample_rate, filename


def resample_audio(audio_data, orig_sr, target_sr):
    """
    Resample audio to target sample rate.

    Args:
        audio_data: Audio samples array
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        numpy array: Resampled audio data
    """
    if orig_sr == target_sr:
        return audio_data

    print(f"🔄 Resampling from {orig_sr}Hz to {target_sr}Hz...")
    return librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)


def transcribe_with_whisper(
    audio_data,
    sample_rate,
    whisper_url="http://whisper-asr:8000",
    beam_size=5,
    vad_filter=True,
    language="en",
):
    """
    Transcribe audio using Whisper ASR service.

    Args:
        audio_data: Audio samples (float32, mono)
        sample_rate: Sample rate of audio
        whisper_url: URL of Whisper service
        beam_size: Beam size for decoding (default: 5, higher = more accurate but slower)
        vad_filter: Enable voice activity detection (default: True, improves accuracy)
        language: Language code (default: 'en')

    Returns:
        str: Transcribed text
    """
    print("🎙️ Transcribing with Whisper...")

    try:
        # Health check
        response = requests.get(f"{whisper_url}/health", timeout=5)
        if response.status_code != 200:
            raise Exception(f"Whisper service not healthy: {response.status_code}")

        # Convert to int16 WAV format
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_int16, sample_rate, format="WAV", subtype="PCM_16")
        wav_buffer.seek(0)
        wav_bytes = wav_buffer.read()

        print(
            f"📤 Uploading {len(audio_data)/sample_rate:.1f}s audio ({len(wav_bytes)/1024:.1f} KB)..."
        )
        print(f"⚙️ Settings: beam_size={beam_size}, vad_filter={vad_filter}, language={language}")

        # Prepare transcription parameters
        data = {"beam_size": beam_size, "vad_filter": vad_filter, "language": language}

        # Send for transcription with longer timeout for large files
        timeout = max(300, int(len(audio_data) / sample_rate * 2))  # At least 2x audio duration
        response = requests.post(
            f"{whisper_url}/transcribe",
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data=data,
            timeout=timeout,
        )

        if response.status_code == 200:
            result = response.json()
            text = result.get("text", "").strip()
            print(f"✅ Transcription completed! ({len(text)} characters)")
            return text
        else:
            error_detail = response.text if response.text else "No error details"
            raise Exception(f"Transcription failed: {response.status_code} - {error_detail}")

    except Exception as e:
        import traceback

        print(f"❌ Error during transcription: {str(e)}")
        traceback.print_exc()
        return ""


def upload_and_process_audio(file_upload, target_sample_rate=16000, transcribe=True):
    """
    Complete workflow: process uploaded audio, resample, and transcribe.

    Args:
        file_upload: FileUpload widget with uploaded file
        target_sample_rate: Target sample rate for processing
        transcribe: Whether to transcribe with Whisper

    Returns:
        tuple: (audio_data, text) for benchmarking
               - audio_data: Processed audio at target sample rate
               - text: Reference transcript (empty string if transcribe=False)
    """
    # Process uploaded file
    audio_data, sample_rate, filename = process_uploaded_file(file_upload)

    # Resample if needed
    audio_data = resample_audio(audio_data, sample_rate, target_sample_rate)

    print(f"✅ Audio ready: {len(audio_data)/target_sample_rate:.2f}s at {target_sample_rate}Hz")

    # Transcribe
    text = ""
    if transcribe:
        text = transcribe_with_whisper(audio_data, target_sample_rate)

    # Display audio player
    print(f"\n🔊 Audio Playback")
    display(Audio(data=audio_data, rate=target_sample_rate, autoplay=False))

    # Display transcript
    if text:
        print(f"\n📄 Reference Transcript")
        print(f"\n{text}")

    print(f"\n✅ Ready for benchmarking!")

    return audio_data, text
