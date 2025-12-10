# Whisper Turbo - GPU ASR Service
# China mirrors enabled by default for faster downloads.

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

# Configure pip for China mirrors (use build args)
RUN pip config set global.index-url ${PYPI_MIRROR} && \
    pip config set global.trusted-host ${PYPI_TRUSTED_HOST}

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set HuggingFace mirror for China (use build arg)
ENV HF_ENDPOINT=${HF_ENDPOINT}

# Install PyTorch with CUDA 12.8 support (cached layer)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision torchaudio --index-url ${PYTORCH_CUDA_MIRROR}

# Install dependencies (cached layer)
COPY requirements.docker.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.docker.txt

# Install flash-attn for faster inference (optional, may fail on some GPUs)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install flash-attn --no-build-isolation || echo "Flash Attention not available"

# Copy shared modules from additional context
COPY --from=shared . /app/shared/

# Copy application code
COPY model.py whisper_service.py ./

# Copy download script LAST (changes here won't invalidate above layers)
COPY download_models.sh .

# Model is pre-downloaded to the cache volume (whisper-hf-cache)
# No download happens during Docker build or runtime - all network requests avoided
# Use download_models.sh for initial setup

EXPOSE 8000

CMD ["python", "whisper_service.py"]
