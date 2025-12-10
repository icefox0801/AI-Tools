# NVIDIA NeMo Parakeet ASR Service
# GPU-accelerated speech recognition with Parakeet CTC/RNNT models
# Supports RTX 5070 Ti (Blackwell CC 12.0) via PyTorch 2.9+ with CUDA 12.8
#
# Uses volume mounts for:
#   - /root/.cache/huggingface  (HuggingFace models)
#   - /root/.cache/nemo         (NeMo models)
#
# China mirrors enabled by default for faster downloads.

# Use NVIDIA CUDA base image with CUDA 12.9 runtime
FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

# Build arguments for mirror configuration (passed from docker-compose.yaml)
ARG APT_MIRROR
ARG PYPI_MIRROR
ARG PYPI_TRUSTED_HOST
ARG HF_ENDPOINT
ARG PYTORCH_CUDA_MIRROR

WORKDIR /app

# Configure APT mirrors for China (use build arg)
RUN sed -i "s|archive.ubuntu.com|${APT_MIRROR}|g" /etc/apt/sources.list && \
    sed -i "s|security.ubuntu.com|${APT_MIRROR}|g" /etc/apt/sources.list

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

# Configure pip for China mirrors (use build args)
RUN pip config set global.index-url ${PYPI_MIRROR} && \
    pip config set global.trusted-host ${PYPI_TRUSTED_HOST}

# Upgrade pip to avoid resolver bugs in system pip
RUN pip install --upgrade pip setuptools wheel

# Set HuggingFace mirror for China (use build arg)
ENV HF_ENDPOINT=${HF_ENDPOINT}

# Install PyTorch 2.9+ with CUDA 12.8 support (compatible with driver 576.88)
# Cache mount stores downloads permanently in Docker's buildkit cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision torchaudio --index-url ${PYTORCH_CUDA_MIRROR}

# Copy ONLY requirements first for better layer caching
# (Code changes won't invalidate pip install layer)
COPY requirements.docker.txt .

# Install Python dependencies
# Note: Installing in stages to avoid resolver issues with complex NeMo deps
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install Cython && pip install -r requirements.docker.txt

# Copy shared modules from additional context
COPY --from=shared . /app/shared/

# Copy application code
COPY parakeet_service.py audio.py parakeet_model.py ./

# Copy download script LAST (changes here won't invalidate above layers)
COPY download_models.sh .

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
