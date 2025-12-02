FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY audio_summary.py .

# Expose port
EXPOSE 7860

# Environment variables
ENV WHISPER_URL=http://whisper-asr:8000
ENV OLLAMA_URL=http://ollama:11434
ENV OLLAMA_MODEL=llama3.2

# Run the application
CMD ["python", "audio_summary.py", "--host", "0.0.0.0", "--port", "7860"]
