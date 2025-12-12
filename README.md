# AI-Tools

Docker-based AI toolkit for real-time speech-to-text, audio transcription, and LLM-powered summarization.

**AI-Tools** is a comprehensive toolkit for professionals, content creators, and developers who need real-time transcription, automated summarization, and live captions—all running locally with no cloud dependencies.

## ✨ What You Can Do

- 🎤 **Live Captions**: Real-time speech-to-text overlay for meetings, videos, and calls
- 📝 **Transcription**: Convert audio to searchable text with high accuracy
- 🤖 **AI Summarization**: Automatically extract key points and create summaries
- 💬 **Chat & Analysis**: Ask questions about your transcripts using local LLMs
- 🔒 **100% Local**: No cloud APIs, no data sent elsewhere, full privacy

---

## 🎬 Getting Started: Your First Workflow

### Setup (One-Time)

```bash
# Start all services
docker compose up -d audio-notes ollama

# Or selectively start services
docker compose up -d vosk-asr whisper-asr parakeet-asr
```

Then access:
- **Audio Notes Web UI**: http://localhost:7860
- **Lobe Chat (LLM Chat)**: http://localhost:3210

---

## 📚 Real-World Workflows

### Workflow 1: Capture a Live Meeting with Captions & Notes

**Live Captions Settings**

<img width="291" height="242" alt="Live Captions Tray Menu" src="https://github.com/user-attachments/assets/45f07f39-fe52-47bc-abb7-d103326087e2" />

**Steps:**

1. **Start Live Captions** (5 minutes before meeting)
   ```bash
   cd apps/live-captions
   python live_captions.py --system-audio --backend whisper
   ```
   - The tray menu appears (see above)
   - Select "Audio Source: System Audio"
   - Select "ASR Model: Whisper" for high accuracy
   - Enable "Recording" to save audio
   - Enable "Auto-Start at Login"

2. **During the Meeting** (Real-time)
   - Caption window shows "Speak now..." prompt at corner of screen
   - Live transcription appears as attendees speak
   - No interruptions to your workflow

3. **After the Meeting** (Post-processing)
   - Open Audio Notes: http://localhost:7860
   - Your recording appears in the Recordings list
   - Click "Transcribe" → Whisper backend generates full transcript
   - Click "Summarize" → AI extracts 3-5 minute summary
   - Use "Chat" tab to ask: "What are the action items?" or "Who owns Task X?"

4. **Export & Share**
   - Download transcript (.txt) 
   - Download subtitles (.srt) for sharing
   - Archive summary as meeting notes

**Result:** 60-minute meeting → 5-minute summary + searchable transcript + documented action items

---

### Workflow 2: Transcribe & Analyze Audio File

**Demo Video with Live Captions**

<img width="1962" height="1124" alt="Demo video with human speech" src="https://github.com/user-attachments/assets/c5cf379d-c510-45cc-a73e-254e30cb6f1b" />

**Steps:**

1. **Open Audio Notes Web UI**
   - Navigate to http://localhost:7860
   - Click "Upload Audio" section

2. **Upload Your File**
   - Select podcast, lecture, interview, or any audio file
   - Supported formats: .mp3, .wav, .m4a, .ogg, etc.

3. **Choose Backend & Transcribe**
   - **For high accuracy**: Choose "Whisper"
   - **For speed**: Choose "Parakeet"
   - Click "Transcribe" button
   - Wait for processing (depends on file length and backend)

4. **Review & Edit**
   - View "Full Transcript" tab
   - Text is fully searchable
   - Edit any misrecognized words directly

5. **Generate Insights**
   - Click "Summary" tab → Get AI summary
   - Click "Chat" tab → Ask specific questions
   - Chat examples:
     - "What are the main topics?"
     - "Summarize the Q&A section"
     - "Create a study guide"

6. **Export**
   - Download transcript as .txt
   - Download subtitles as .srt
   - Share summary with team

**Result:** 2-hour podcast → 30 min Whisper transcription + 2 min AI summary + instant Q&A

---

### Workflow 3: Real-Time Caption Overlay for Content Review

**Live Transcription in Progress**

<img width="3080" height="268" alt="Speak now prompt showing live transcription" src="https://github.com/user-attachments/assets/7c3d5e5e-6827-42e1-a6c6-86f944837721" />

**Scenario:** You're watching video content and want live captions to check audio quality.

**Steps:**

1. **Launch Live Captions**
   ```bash
   python live_captions.py --system-audio --backend vosk
   ```
   - Use "Vosk" for fast, real-time captions (CPU-based)
   - Window docks in corner, doesn't block content

2. **Play Your Video**
   - Open video in your player
   - Live Captions shows real-time transcription
   - Monitor for:
     - Unclear speech
     - Audio quality issues
     - Background noise
     - Timing problems

3. **Mark Problem Areas**
   - When you spot an issue, note the timestamp
   - Enable "Recording" to save the problematic segment
   - Use captions to identify exact words

4. **Re-Record & Fix**
   - Re-record the problematic section
   - Upload to Audio Notes
   - Transcribe again with Whisper
   - Compare against original

5. **Generate Final Captions**
   - Export as .srt subtitle file
   - Use in video editor
   - Upload to platform with confidence

**Result:** Video with audio quality verified + accurate subtitle file + confidence in final output

---

### Workflow 4: Research & Knowledge Extraction

**Audio Notes Analysis Interface**

<img width="2848" height="982" alt="Audio Notes with Full Transcript, Summary, and Chat tabs" src="https://github.com/user-attachments/assets/47fbc1b1-70c2-45fd-885c-ca000714c1d7" />

**Scenario:** You're researching a topic with multiple sources (lectures, interviews, podcasts).

**Steps:**

1. **Batch Upload Sources**
   - Audio Notes: Upload 5-10 recordings
   - Queue all for transcription
   - Whisper processes in parallel

2. **Create Research Index**
   - Each transcript becomes searchable
   - Summarize each source
   - Take notes on relationships

3. **Extract Cross-Source Insights**
   - Use Chat to ask:
     - "Compare these 3 sources on AI ethics"
     - "What are the top 10 concepts across all lectures?"
     - "Create a study guide combining all sources"
   - LLM synthesizes answers from all transcripts

4. **Build Knowledge Base**
   - Export all summaries as markdown
   - Create bibliography with timestamps
   - Organize into learning modules
   - Link related concepts across sources

5. **Document Findings**
   - Create comprehensive research report
   - Include direct quotes from transcripts
   - Reference time codes to original audio
   - Share with study group or team

**Result:** 10 hours of audio → Organized research notes + cross-source analysis + study materials (in 30 min)

---

## 🔧 Configuration & Customization

### Choose Your ASR Backend

| Backend | Speed | Accuracy | Languages | GPU | Best For |
|---------|-------|----------|-----------|-----|----------|
| **Vosk** | ⚡ Fast | ⭐⭐ Good | English | ❌ | Real-time captions, Live events |
| **Parakeet** | ⚡ Fast | ⭐⭐⭐⭐ Excellent | English | ✅ | Meetings, Professional use |
| **Whisper** | 🐢 Slower | ⭐⭐⭐⭐ Excellent | 99 Languages | ✅ | Final transcripts, Multilingual |

**In Audio Notes:**
- Click your preferred backend button before transcribing
- Default is Whisper (highest quality)

**In Live Captions:**
```bash
python live_captions.py --backend vosk      # Real-time, lightweight
python live_captions.py --backend whisper   # High accuracy, GPU required
python live_captions.py --backend parakeet  # Fast + accurate
```

### Setup Local LLM for AI Features

Audio Notes uses local LLMs (Ollama) for summarization and chat.

1. **Start Ollama**
   ```bash
   docker compose up -d ollama
   ```

2. **Pull a Model**
   ```bash
   docker exec ollama ollama pull qwen:7b      # Recommended
   docker exec ollama ollama pull llama2:7b    # Alternative
   ```

3. **Access Chat UI**
   - Lobe Chat: http://localhost:3210
   - Or use chat in Audio Notes web UI

**Recommended Models:**
- **qwen:7b** - Fast, good quality, multilingual
- **llama2:7b** - General purpose, good for summarization
- **mistral:7b** - Creative, good for analysis
python live_captions.py --debug
```

See [apps/live-captions/README.md](apps/live-captions/README.md) for details.

### How to Use Live Captions

1. **Installation:**
   ```bash
   cd apps/live-captions
   pip install -r requirements.txt
   ```

2. **Start Live Captions:**
   - **System audio** (capture screen/speaker audio):
     ```bash
     python live_captions.py --system-audio --backend whisper
     ```
   
   - **Microphone** (capture from mic):
     ```bash
     python live_captions.py --backend vosk  # Fast, CPU
     ```

3. **Run the tray application:**
   ```bash
   # Build and run the compiled .exe
   python -m PyInstaller live_captions.spec
   ./dist/Live\ Captions.exe
   ```

4. **Using Live Captions:**
   - Window appears in the corner of your screen
   - Shows real-time transcription as you speak or watch videos
   - Choose backend for best performance:
     - **Vosk**: CPU-based, fast, lightweight (good for microphone)
     - **Parakeet**: GPU-based, highest accuracy (needs NVIDIA GPU)
     - **Whisper**: GPU-based, best for clarity (needs NVIDIA GPU)

5. **Sync with Audio Notes:**
   - Enable "Save to Audio Notes" in settings
   - Transcripts automatically appear in the web app
   - Refine and summarize there

**Use cases:**
- Accessibility during online classes or meetings
- Subtitle generation for content review
- Real-time note-taking during presentations
- Verify audio quality before final transcription

## 📊 System Architecture

```
+-------------------+       WebSocket       +---------------------+
|   Client Apps     |  ---------------->    |    ASR Services     |
|                   |     Audio Stream      |    (Docker)         |
| - Audio Notes     |  <----------------    |  - Vosk (CPU)       |
|   (Web UI)        |     Transcripts       |  - Parakeet (GPU)   |
| - Live Captions   |                       |  - Whisper (GPU)    |
|   (Desktop)       |                       +---------------------+
+-------------------+                                |
        |                                            v
        |                               +---------------------+
        +-----------------------------> |   Ollama LLM        |
              Chat & Summarize          |   (Local Models)    |
                                        +---------------------+
```

## 🐳 Available Services

| Service | Port | Purpose |
|---------|------|---------|
| `audio-notes` | 7860 | Web UI for transcription & AI analysis |
| `whisper-asr` | 8003 | OpenAI Whisper (multilingual, high accuracy) |
| `parakeet-asr` | 8002 | NVIDIA Parakeet (fast, English, GPU) |
| `vosk-asr` | 8001 | Vosk (real-time, CPU, lightweight) |
| `text-refiner` | 8010 | Punctuation & error correction |
| `ollama` | 11434 | Local LLM runtime (for summarization) |
| `lobe-chat` | 3210 | Chat UI for Ollama |

## 📁 Project Structure

```
AI-Tools/
├── apps/live-captions/       # Desktop tray app for live captions
├── services/
│   ├── audio-notes/          # Web UI (Flask + frontend)
│   ├── whisper/              # Whisper ASR service
│   ├── parakeet/             # Parakeet ASR service
│   ├── vosk/                 # Vosk ASR service
│   └── text-refiner/         # Text post-processing
├── shared/                   # Common utilities
├── integration/              # End-to-end tests
└── docker-compose.yaml       # Service orchestration
```

## 🧪 Testing & Development

```bash
# Run all tests
python -m pytest apps/ services/ shared/ integration/ -v

# Run with coverage
python -m pytest --cov=apps --cov=services --cov=shared

# Format code
black apps/ services/ shared/

# Lint code
ruff check apps/ services/ shared/
```

## 🤝 Contributing

Contributions welcome! Areas needing help:

- Additional ASR backends (Google Cloud Speech, Azure, etc.)
- Language-specific models and tuning
- Mobile app for captions
- Cloud deployment guides
- Documentation improvements

## 📄 License

MIT License - see LICENSE file for details
   ├─ Use timestamps from captions to find problem areas
   ├─ Re-record problematic sections
   ├─ Upload corrected audio to Audio Notes
   ├─ Transcribe again with Whisper for final version
   └─ Compare against original transcript

4. Final Output
   ├─ Export clean transcript
   ├─ Generate captions/subtitles (.srt)
   └─ Use in video editor or upload to platform
```

**Benefit:** Catch audio problems before final publication, ensure accessibility through accurate captions

---

### Workflow 4: Research & Knowledge Extraction

**Scenario:** You're researching a topic with multiple audio sources (lectures, interviews, podcasts) and need organized notes.

```
1. Batch Upload
   ├─ Audio Notes: Upload 5 lecture recordings
   ├─ Queue transcription for all files
   ├─ Whisper processes in parallel
   └─ Wait 15-20 minutes for batch completion

2. Create Research Index
   ├─ Each transcript becomes searchable
   ├─ Summarize each source:
   │  ├─ Lecture 1: "Basic concepts of machine learning"
   │  ├─ Lecture 2: "Neural networks and deep learning"
   │  ├─ Lecture 3: "Practical applications of AI"
   │  └─ Interview 1: "Industry perspectives on AI ethics"
   └─ Chat: "Summarize the key differences between these sources"

3. Extract Knowledge
   ├─ Chat tab queries:
   │  ├─ "What are the top 10 concepts discussed?"
   │  ├─ "How do these sources differ in their views?"
   │  └─ "Create a study guide from all transcripts"
   ├─ AI generates:
   │  ├─ Comparative analysis
   │  ├─ Key learning points
   │  └─ Study materials
   └─ Cross-reference timestamps to original audio

4. Document Findings
   ├─ Export summaries as markdown notes
   ├─ Create bibliography with sources
   ├─ Build knowledge base from transcripts
   └─ Use as reference material
```

**Outcome:** 10 hours of audio content → Organized notes, summaries, and study materials in 30 minutes

---

### Quick Reference: Choosing Your Path

| Your Situation | Best Tool | Configuration |
|---|---|---|
| **Live meeting/call** | Live Captions + Audio Notes | System Audio + Whisper |
| **Podcast/lecture recording** | Audio Notes | Upload + Whisper |
| **Real-time accessibility** | Live Captions | Microphone + Vosk |
| **High-volume transcription** | Audio Notes batch mode | Upload multiple + Parakeet |
| **Video caption generation** | Live Captions | System Audio + Vosk (fast) |
| **Research/knowledge extraction** | Audio Notes Chat | Upload + Whisper + LLM |

---

## Configuration

Select ASR backend in `shared/config/backends.py`:

```python
BACKEND = "vosk"      # CPU-based, lightweight
# BACKEND = "parakeet"  # GPU-based, high accuracy
# BACKEND = "whisper"   # GPU-based, multilingual
```

Or use command line:
```bash
python live_captions.py --backend whisper --system-audio
```

### Choosing the Right ASR Backend

| Backend | Performance | Accuracy | Language Support | GPU Required | CPU Cost |
|---------|-------------|----------|------------------|--------------|----------|
| **Vosk** | ⚡ Fast | ⭐⭐ Good | English, limited | ❌ No | Low |
| **Parakeet** | ⚡ Fast | ⭐⭐⭐⭐ Excellent | English | ✅ Yes (RTX 3060+) | Medium |
| **Whisper** | 🐢 Slower | ⭐⭐⭐⭐ Excellent | 99 Languages | ✅ Yes (RTX 3060+) | High |

**Recommendations:**
- **Real-time live captions?** → Use Vosk (CPU) or Parakeet (GPU)
- **High accuracy needed?** → Use Whisper or Parakeet
- **Multilingual support?** → Use Whisper
- **CPU-only system?** → Use Vosk
- **Maximum accuracy + speed?** → Use Parakeet (GPU required)

### Configuring Local LLM

Audio Notes can summarize and chat using local LLMs via Ollama.

1. **Start Ollama:**
   ```bash
   docker compose up -d ollama
   ```

2. **Pull a model:**
   ```bash
   docker exec ollama ollama pull qwen:7b  # Fast, good quality
   docker exec ollama ollama pull llama2:7b  # Alternative
   ```

3. **Access LLM chat UI:**
   - Lobe Chat: http://localhost:3210
   - Audio Notes summarization uses these models automatically

**Recommended models:**
- **qwen:7b** - Fast, good English/Chinese support
- **llama2:7b** - General purpose, good for summarization
- **mistral:7b** - Creative, good for analysis




