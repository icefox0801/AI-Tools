# Vosk ASR Service
# Lightweight, fast, offline speech recognition
# China mirrors enabled by default for faster downloads.

FROM python:3.11-slim

# Build arguments for mirror configuration (passed from docker-compose.yaml)
ARG APT_MIRROR
ARG PYPI_MIRROR
ARG PYPI_TRUSTED_HOST
ARG HF_ENDPOINT

WORKDIR /app

# Configure APT mirrors for China (use build arg)
RUN sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list 2>/dev/null || true

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Configure pip for China mirrors (use build args)
RUN pip config set global.index-url ${PYPI_MIRROR} && \
    pip config set global.trusted-host ${PYPI_TRUSTED_HOST}

# Set HuggingFace mirror for China (use build arg)
ENV HF_ENDPOINT=${HF_ENDPOINT}

# Install Python dependencies
COPY requirements.docker.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.docker.txt

# Copy service code
COPY vosk_service.py .
COPY download_models.sh .
RUN chmod +x download_models.sh

# Copy shared modules from additional context
COPY --from=shared . /app/shared/

# Model will be mounted as volume at /app/model
# Run download_models.sh manually before starting service

EXPOSE 8000

CMD ["python", "vosk_service.py"]
