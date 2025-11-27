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

# Copy service code (includes model download script)
COPY vosk_service.py .
COPY download_model.sh .
RUN chmod +x download_model.sh

# Model will be mounted as volume at /app/model
# Download script runs on startup if model is missing

EXPOSE 8000

CMD ["./download_model.sh"]
