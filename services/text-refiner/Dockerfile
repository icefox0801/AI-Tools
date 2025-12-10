# Text Refiner Service
# Punctuation restoration and ASR error correction
# GPU-accelerated T5 for correction, ONNX for punctuation
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

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
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

# Install PyTorch with CUDA support
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision torchaudio --index-url ${PYTORCH_CUDA_MIRROR}

# Copy requirements
COPY requirements.docker.txt .

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.docker.txt

# Pre-download models during build for faster startup
# Punctuation model (~100MB)
RUN python -c "from punctuators.models import PunctCapSegModelONNX; PunctCapSegModelONNX.from_pretrained('pcs_en')"

# Spelling Correction model (~500MB) - uses BART architecture
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    AutoTokenizer.from_pretrained('oliverguhr/spelling-correction-english-base'); \
    AutoModelForSeq2SeqLM.from_pretrained('oliverguhr/spelling-correction-english-base')"

# Copy shared modules from additional context
COPY --from=shared . /app/shared/

# Copy application code
COPY text_refiner_service.py ./

# Copy download script LAST
COPY download_models.sh .

EXPOSE 8000

CMD ["python", "text_refiner_service.py"]
