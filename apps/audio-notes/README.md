# Audio Notes

A Gradio web UI for audio transcription and AI-powered summarization.

## Features

- 🎙️ **Full Transcript** - Complete text transcription via Whisper ASR
- 📋 **AI Summary** - Key points and overview via Ollama LLM
- 💬 **Interactive Chat** - Ask questions about the content

## Prerequisites

Ensure these services are running (via Docker Compose):
- **Whisper ASR** - `http://localhost:8003`
- **Ollama** - `http://localhost:11434` with `llama3.2` model

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python audio_notes.py
```

### With Audio File
```bash
python audio_notes.py --audio recording.wav
```

### Custom Port
```bash
python audio_notes.py --port 8080
```

### All Options
```bash
python audio_notes.py --help
```

## Integration with Live Captions

This app works with the Live Captions system tray app:

1. Enable recording in Live Captions (🔴 Recording: ON)
2. Start streaming audio
3. When done, click "📝 Transcribe & Summarize" in the tray menu
4. Audio Notes opens with your recording

## Configuration

Environment variables:
- `WHISPER_URL` - Whisper ASR endpoint (default: `http://localhost:8003`)
- `OLLAMA_URL` - Ollama endpoint (default: `http://localhost:11434`)
- `OLLAMA_MODEL` - LLM model to use (default: `llama3.2`)

## Screenshots

```
📝 Audio Notes
├── 🎵 Upload or record audio
├── 📋 Get AI-generated summary
├── 📜 View full transcript
└── 💬 Chat about the content
```
