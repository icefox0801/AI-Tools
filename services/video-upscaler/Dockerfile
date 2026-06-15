# Video Upscaler Service - Real-ESRGAN GPU video super-resolution
# Increases video clarity / resolution for offline batch processing.
#
# Supports RTX 5070 Ti (Blackwell CC 12.0 / sm_120) via PyTorch 2.9+ with CUDA 12.8.
# Real-ESRGAN's network (RRDBNet / SRVGG) is pure PyTorch, so it runs reliably on
# Blackwell GPUs - unlike mmediting-based VSR models which cannot build on torch 2.9.
#
# Uses volume mounts for:
#   - /app/weights  (Real-ESRGAN model weights, downloaded once)
#   - /app/data     (uploaded inputs + upscaled outputs)
#
# China mirrors enabled by default for faster downloads.

# Use NVIDIA CUDA base image with CUDA 12.9 runtime (matches other GPU services)
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

# Install Python, FFmpeg (frame extract/reassemble) and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    wget \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Configure pip for China mirrors (use build args)
RUN pip config set global.index-url ${PYPI_MIRROR} && \
    pip config set global.trusted-host ${PYPI_TRUSTED_HOST}

# Upgrade pip to avoid resolver bugs in system pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch 2.9.1 + torchvision 0.24.1 with CUDA 12.8 (Blackwell sm_120 support).
# torch 2.9.x pairs with torchvision 0.24.x.
# IMPORTANT: use ONLY the PyTorch index here (no PyPI extra-index). The cu128 index is
# self-contained and co-locates the large NVIDIA CUDA wheels (cudnn/cublas/nvrtc, ~2GB).
# Adding a PyPI extra-index makes pip pull those NVIDIA wheels from a slow PyPI mirror.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.9.1 torchvision==0.24.1 --index-url ${PYTORCH_CUDA_MIRROR}

# Install application dependencies (realesrgan, basicsr, fastapi, opencv, ...).
# basicsr builds from an sdist and does `import torch` at the top of setup.py, so
# under pip's isolated build it would download a SECOND (CPU) torch from PyPI and
# appear to hang. Pre-install Cython (basicsr build dep) and disable build
# isolation so basicsr reuses the GPU torch already installed above.
COPY requirements.docker.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install Cython && \
    pip install --no-build-isolation -r requirements.docker.txt

# Patch basicsr for torchvision >= 0.17 (functional_tensor was removed/moved).
# basicsr/data/degradations.py imports rgb_to_grayscale from the old location.
# NOTE: we must NOT import basicsr to locate the file (the import itself fails on
# the broken line); instead find the file on disk and patch it in place.
RUN DEG="$(find / -path '*/basicsr/data/degradations.py' -print -quit)" && \
    test -n "$DEG" && \
    sed -i 's/from torchvision.transforms.functional_tensor import/from torchvision.transforms.functional import/' "$DEG" && \
    python -c "import basicsr.data.degradations" && \
    echo "Patched: $DEG"

# Copy application code
COPY log_setup.py upscaler_model.py pipeline.py jobs.py upscaler_service.py ./

# Copy weight download script LAST (changes here won't invalidate above layers)
COPY download_models.sh .

# Environment defaults
ENV DEVICE=cuda
ENV WEIGHTS_DIR=/app/weights
ENV DATA_DIR=/app/data
ENV UPSCALER_TILE=0
ENV UPSCALER_HALF=true
ENV PYTHONUNBUFFERED=1

# Pre-create runtime directories
RUN mkdir -p /app/weights /app/data/inputs /app/data/outputs

# Expose FastAPI port
EXPOSE 8000

# Health check (model loads lazily on first job)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI service
CMD ["python3", "upscaler_service.py"]
