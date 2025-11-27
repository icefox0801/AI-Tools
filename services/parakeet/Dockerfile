# NVIDIA NeMo Parakeet ASR Service
# GPU-accelerated speech recognition with Parakeet CTC/RNNT models
# Supports RTX 5070 Ti (Blackwell CC 12.0) via PyTorch 2.9+ with CUDA 12.8
#
# Uses volume mounts for:
#   - /root/.cache/huggingface  (HuggingFace models)
#   - /root/.cache/nemo         (NeMo models)

# Use NVIDIA CUDA base image with CUDA 12.9 runtime
FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    curl \
    gcc \
    g++ \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to avoid resolver bugs in system pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch 2.9+ with CUDA 12.8 support (compatible with driver 576.88)
# Using BuildKit cache mount to persist pip downloads across builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Copy requirements first for better layer caching
COPY requirements.docker.txt .

# Install Python dependencies
# Note: Installing in stages to avoid resolver issues with complex NeMo deps
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install Cython && pip install -r requirements.docker.txt

# Copy application code
COPY parakeet_service.py .

# Expose port
EXPOSE 8000

# Workaround: Disable TorchScript JIT to avoid SIGSEGV in NeMo's online_clustering.py
# This is a known issue with NeMo 2.5.x and PyTorch 2.9.x
ENV PYTORCH_JIT=0

# Health check with longer start period for first-time model download
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the service
CMD ["python", "parakeet_service.py"]
