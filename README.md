# AI-Tools

Docker-based AI toolkit for real-time speech-to-text, audio transcription, and LLM-powered summarization—all running locally with no cloud dependencies.

## ✨ Features

- 🎤 **Live Captions** - Real-time speech-to-text overlay for meetings, videos, and calls
- 📝 **Transcription** - Convert audio to searchable text with high accuracy
- 🤖 **AI Summarization** - Automatically extract key points and create summaries
- 💬 **Chat with Transcripts** - Ask questions about your content using local LLMs
- 🔒 **100% Local** - No cloud APIs, no data sent elsewhere, full privacy

---

## 🚀 Quick Start

```bash
# Start all services
docker compose up -d audio-notes ollama whisper-asr

# Access the web UI
open http://localhost:7860
```

---

## 💻 Hardware & Platform Requirements

**Platform**: Windows 10/11 (required for Live Captions desktop app)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16+ GB |
| **GPU** | - | NVIDIA RTX 3060+ (8GB VRAM) |
| **Storage** | 20 GB | 50+ GB (for models) |

**GPU Notes:**
- **Vosk**: CPU only, works on any system
- **Parakeet/Whisper**: Requires NVIDIA GPU with CUDA support
- VRAM usage: ~4-6GB per ASR model loaded
- Multiple GPU models can run simultaneously with 12GB+ VRAM

---

## 🎬 Complete Workflow: From Audio to Insights

This is the typical workflow showing how Live Captions and Audio Notes work together.

### Step 1: Configure Live Captions

Launch Live Captions from system tray and configure your settings:

<img width="291" alt="Live Captions Tray Menu" src="https://github.com/user-attachments/assets/45f07f39-fe52-47bc-abb7-d103326087e2" />

- **Audio Source**: Choose "System Audio" (for videos/meetings) or "Microphone" (for your voice)
- **ASR Model**: Select Whisper (accurate), Parakeet (fast), or Vosk (CPU)
- **Enable Recording**: Turn on to save audio for later transcription
- **Live Transcription**: Enable to see real-time captions

### Step 2: Capture Audio

Play a video, join a meeting, or speak into your microphone:

<img width="1962" alt="Demo video with human speech" src="https://github.com/user-attachments/assets/c5cf379d-c510-45cc-a73e-254e30cb6f1b" />

Live Captions displays real-time transcription as audio plays:

<img width="3080" alt="Speak now - Live transcription in progress" src="https://github.com/user-attachments/assets/7c3d5e5e-6827-42e1-a6c6-86f944837721" />

- Captions appear in a floating overlay window
- The window shows "Speak now..." when ready to capture
- Recording saves audio files automatically when enabled

### Step 3: Transcribe & Analyze in Audio Notes

Open Audio Notes at http://localhost:7860 to process your recordings:

<img width="2848" alt="Audio Notes with Full Transcript, Summary, and Chat" src="https://github.com/user-attachments/assets/47fbc1b1-70c2-45fd-885c-ca000714c1d7" />

1. **Select Recording** - Your saved recordings appear in the Recordings panel
2. **Choose ASR Backend** - Whisper (accurate) or Parakeet (fast)
3. **Click Transcribe** - Generates full text transcript
4. **View Tabs**:
   - **Full Transcript** - Complete searchable text
   - **Summary** - AI-generated summary of key points
   - **Chat** - Ask questions about the content

### Step 4: Chat with Your Content

Use the Chat tab to interact with your transcript:

- "What are the main topics discussed?"
- "Summarize the key action items"
- "What did the speaker say about machine learning?"
- "Create a study guide from this content"

**Result**: Audio → Real-time captions → Saved recording → Transcript → Summary → Interactive Q&A

---

## 🔧 ASR Backend Comparison

| Backend | Speed | Accuracy | Languages | GPU Required | Best For |
|---------|-------|----------|-----------|--------------|----------|
| **Vosk** | ⚡ Fast | Good | English | No | Real-time captions |
| **Parakeet** | ⚡ Fast | Excellent | English | Yes | Meetings, speed |
| **Whisper** | Slower | Excellent | 99+ | Yes | Final transcripts |

---

## 🤖 Model Architecture Details

| Service | Model | Architecture | Parameters | VRAM | Features |
|---------|-------|--------------|------------|------|----------|
| **Whisper** | openai/whisper-large-v3-turbo | Transformer (Seq2Seq) | 809M | 1.5GB | Multilingual (99+ languages), native punctuation, timestamps |
| **FastConformer** | nvidia/stt_en_fastconformer_hybrid_large_streaming_multi | FastConformer (Hybrid RNNT/CTC) | 114M | 2-3GB | Low-latency streaming, cache-aware, native punctuation, English-only |
| **Parakeet (Streaming)** | nvidia/parakeet-tdt-1.1b | Conformer (TDT-CTC) | 1.1B | 2.0GB | Cache-aware encoder, incremental decoding, English-only |
| **Parakeet (Offline)** | nvidia/parakeet-rnnt-1.1b | Conformer (RNN-T) | 1.1B | 4.1GB | Higher accuracy, better for final transcripts, English-only |
| **Vosk** | vosk-model-en-us-0.22 | TDNN-LSTM | ~40M | CPU only | Lightweight, offline, real-time capable |
| **VAD (Optional)** | silero_vad | LSTM | ~2M | ~50MB | Voice activity detection, skip silence |
| **Text Refiner** | oliverguhr/spelling-correction-english-base | BERT | 110M | ~500MB | Punctuation & spelling correction |

**Notes:**
- **Transformer**: Encoder-decoder architecture (Whisper)
- **FastConformer**: Optimized Conformer with cache-aware streaming (FastConformer)
- **Conformer**: CNN + Transformer hybrid for better audio modeling (Parakeet)
- **TDT-CTC**: Time-delay Transformer with CTC loss for streaming
- **RNN-T**: Recurrent Neural Network Transducer for offline transcription
- **Hybrid RNNT/CTC**: Multitask training with both decoders for better accuracy
- **TDNN-LSTM**: Time-Delay Neural Network with LSTM (Vosk)

**GPU Memory Usage:**
- Single ASR model: 1.5-4GB VRAM (varies by model)
- Running 2 models simultaneously: 3.5-6GB VRAM
- All services + LLM: 10-14GB VRAM recommended

**Recommended for Streaming:**
- **FastConformer**: Best balance of latency and accuracy (WER 5.4%, 240ms avg latency)
- **Parakeet TDT**: Good for cache-aware streaming (WER ~6%, higher latency)
- **Whisper**: Best for multilingual, but higher latency

---

## 🐳 Available Services

| Service | Port | Description |
|---------|------|-------------|
| audio-notes | 7860 | Web UI for transcription & analysis |
| fastconformer-asr | 8004 | NVIDIA FastConformer ASR (streaming-optimized) |
| whisper-asr | 8003 | OpenAI Whisper ASR (multilingual) |
| parakeet-asr | 8002 | NVIDIA Parakeet ASR (fast, English) |
| vosk-asr | 8001 | Vosk ASR (CPU, lightweight) |
| text-refiner | 8010 | Punctuation & error correction |
| ollama | 11434 | Local LLM runtime |
| lobe-chat | 3210 | Chat UI for Ollama |

---

## 📁 Project Structure

```
AI-Tools/
├── apps/live-captions/     # Desktop tray app for live captions
├── services/
│   ├── audio-notes/        # Web UI (Gradio)
│   ├── fastconformer/      # FastConformer ASR service (streaming-only)
│   ├── whisper/            # Whisper ASR service
│   ├── parakeet/           # Parakeet ASR service
│   ├── vosk/               # Vosk ASR service
│   └── text-refiner/       # Text post-processing
├── shared/                 # Common utilities
├── integration/            # End-to-end tests
└── docker-compose.yaml     # Service orchestration
```

---

## 🧪 Development

```bash
# Run all tests
python -m pytest apps/ services/ shared/ integration/ -v

# Run with coverage
python -m pytest --cov=apps --cov=services --cov=shared

# Lint code
ruff check apps/ services/ shared/
```

---

## 📄 License

MIT License - see LICENSE file for details.




