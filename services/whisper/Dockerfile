# Whisper Large V3 Turbo - GPU ASR Service
FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.8 support (cached layer)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install dependencies (cached layer)
COPY requirements.docker.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.docker.txt

# Install flash-attn for faster inference (optional, may fail on some GPUs)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install flash-attn --no-build-isolation || echo "Flash Attention not available"

# Copy application code
COPY whisper_service.py .

# Copy shared modules from additional context
COPY --from=shared . /app/shared/

# Pre-download model (optional - can be done at runtime)
# RUN python -c "from transformers import AutoModelForSpeechSeq2Seq; AutoModelForSpeechSeq2Seq.from_pretrained('openai/whisper-large-v3-turbo')"

EXPOSE 8000

CMD ["python", "whisper_service.py"]
