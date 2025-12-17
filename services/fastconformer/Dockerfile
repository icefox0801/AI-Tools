# NVIDIA FastConformer Hybrid ASR Service
# GPU-accelerated streaming speech recognition optimized for low latency
# Supports RTX 5070 Ti (Blackwell CC 12.0) via PyTorch 2.9+ with CUDA 12.8
# Requires PyTorch 2.9.1+ for Blackwell architecture support (sm_120)
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

# Install PyTorch 2.9.1 with CUDA 12.8 support (compatible with Blackwell sm_120)
# PyTorch 2.9+ is required for RTX 5070 Ti and other Blackwell GPUs
# Cache mount stores downloads permanently in Docker's buildkit cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.9.1 torchaudio==2.9.1 --index-url ${PYTORCH_CUDA_MIRROR} --extra-index-url ${PYPI_MIRROR}

# Install NeMo and dependencies
# First install num2words from PyPI (not in Aliyun mirror)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://pypi.org/simple/ num2words

# Install dependencies from requirements file
COPY requirements.docker.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install Cython && \
    pip install -r requirements.docker.txt

# Copy shared modules from additional context
COPY --from=shared . /app/shared/

# Copy application code
COPY fastconformer_service.py fastconformer_model.py ./

# Copy download script LAST (changes here won't invalidate above layers)
COPY download_models.sh .

# Environment variables for model configuration
ENV FASTCONFORMER_MODEL=nvidia/stt_en_fastconformer_hybrid_large_streaming_multi
ENV DEVICE=cuda
ENV DECODER_TYPE=rnnt
ENV ATT_CONTEXT_SIZE=[70,6]

# Expose port for FastAPI service
EXPOSE 8000

# Workaround: Disable TorchScript JIT to avoid SIGSEGV in NeMo's online_clustering.py
# This is a known issue with NeMo 2.3.x and PyTorch 2.6.x
ENV PYTORCH_JIT=0

# Health check (wait for model to load)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI service
CMD ["python3", "fastconformer_service.py"]
