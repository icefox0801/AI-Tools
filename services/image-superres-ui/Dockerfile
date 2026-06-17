FROM python:3.11-slim

WORKDIR /app

RUN sed -i "s|deb.debian.org|mirrors.aliyun.com|g" /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i "s|deb.debian.org|mirrors.aliyun.com|g" /etc/apt/sources.list 2>/dev/null || true

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.aliyun.com

COPY requirements.docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py app.py ./

EXPOSE 7862

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7862/ || exit 1

ENV IMAGE_SR_URL=http://image-superres:8000
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7862

CMD ["python", "app.py"]
