# Text Refiner Service
# Punctuation restoration and ASR error correction
# GPU-accelerated T5 for correction, ONNX for punctuation

FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

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

# Copy application
COPY text_refiner_service.py .

# Copy shared modules from additional context
COPY --from=shared . /app/shared/

EXPOSE 8000

CMD ["python", "text_refiner_service.py"]
