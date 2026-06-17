FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

ARG APT_MIRROR
ARG PYPI_MIRROR
ARG PYPI_TRUSTED_HOST
ARG PYTORCH_CUDA_MIRROR

WORKDIR /app

RUN sed -i "s|archive.ubuntu.com|${APT_MIRROR}|g" /etc/apt/sources.list && \
    sed -i "s|security.ubuntu.com|${APT_MIRROR}|g" /etc/apt/sources.list

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

RUN pip config set global.index-url ${PYPI_MIRROR} && \
    pip config set global.trusted-host ${PYPI_TRUSTED_HOST}

RUN pip install --upgrade pip setuptools wheel

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.9.1 torchvision==0.24.1 --index-url ${PYTORCH_CUDA_MIRROR}

COPY requirements.docker.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install Cython && \
    pip install --no-build-isolation -r requirements.docker.txt

RUN DEG="$(find / -path '*/basicsr/data/degradations.py' -print -quit)" && \
    test -n "$DEG" && \
    sed -i 's/from torchvision.transforms.functional_tensor import/from torchvision.transforms.functional import/' "$DEG" && \
    python -c "import basicsr.data.degradations"

COPY log_setup.py image_model.py image_generative.py image_service.py ./
COPY download_models.sh .

ENV DEVICE=cuda
ENV WEIGHTS_DIR=/app/weights
ENV DATA_DIR=/app/data
ENV UPSCALER_TILE=0
ENV UPSCALER_HALF=true
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /app/weights /app/data/outputs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "image_service.py"]
