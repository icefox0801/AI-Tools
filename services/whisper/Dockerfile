# Faster-Whisper ASR Service - GPU mode with CUDA
# Use CUDA 12.4 with cuDNN 9 and Python 3.11
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    curl \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (Docker-specific)
COPY requirements.docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY whisper_service.py /app/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "whisper_service:app", "--host", "0.0.0.0", "--port", "8000"]
