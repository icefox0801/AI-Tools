# Faster-Whisper ASR Service - GPU mode with CUDA
# Use CUDA 12.4 with cuDNN 9 for compatibility with faster-whisper 1.0+
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Install faster-whisper with CUDA support
# Pin huggingface_hub to avoid XET download issues
RUN pip install --no-cache-dir \
    huggingface_hub==0.23.0 \
    faster-whisper==1.0.3 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    websockets==12.0 \
    python-multipart==0.0.6 \
    soundfile==0.12.1 \
    numpy==1.24.3 \
    requests

COPY whisper_service.py /app/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "whisper_service:app", "--host", "0.0.0.0", "--port", "8000"]
