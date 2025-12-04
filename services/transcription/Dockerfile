# Transcription Gateway - Lightweight API orchestrator
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

# Copy service code
COPY transcription_service.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run service
CMD ["python", "transcription_service.py"]
