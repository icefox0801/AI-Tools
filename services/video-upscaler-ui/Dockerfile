# Video Upscaler Web UI - Gradio frontend for the video-upscaler API
# Thin client: uploads a video, polls job progress, previews/downloads result.
# China mirrors enabled by default for faster downloads.

FROM python:3.11-slim

WORKDIR /app

# Configure APT mirrors for China (Aliyun)
RUN sed -i "s|deb.debian.org|mirrors.aliyun.com|g" /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i "s|deb.debian.org|mirrors.aliyun.com|g" /etc/apt/sources.list 2>/dev/null || true

# Install system dependencies (ffmpeg for video probing/aspect-ratio detection)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Configure pip for China mirrors (Aliyun)
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.aliyun.com

# Install dependencies first for layer caching
COPY requirements.docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py .
COPY app.py .

# Expose Gradio port
EXPOSE 7861

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7861/ || exit 1

# Environment defaults
ENV UPSCALER_URL=http://video-upscaler:8000
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7861

CMD ["python", "app.py"]
