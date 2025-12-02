# Audio Notes Service - Gradio Web UI for Transcription & Summarization
# This service provides a web interface to:
# 1. Browse recordings from shared volume
# 2. Transcribe audio files using Whisper ASR
# 3. Summarize transcripts using Ollama LLM
# 4. Interactive chat for Q&A about content

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY audio_notes.py .

# Create recordings directory
RUN mkdir -p /app/recordings

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Environment variables with defaults
ENV WHISPER_URL=http://whisper-asr:8000
ENV OLLAMA_URL=http://ollama:11434
ENV OLLAMA_MODEL=qwen3:14b
ENV RECORDINGS_DIR=/app/recordings
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Run the application
CMD ["python", "audio_notes.py", "--host", "0.0.0.0", "--port", "7860"]
