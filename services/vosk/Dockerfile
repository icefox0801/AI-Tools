# Vosk ASR Service
# Lightweight, fast, offline speech recognition

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

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
