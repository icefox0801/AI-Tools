# Use the official Jupyter PySpark notebook as base image
# Note: For China users, consider using a Docker Hub mirror in daemon.json
# or pushing quay.io/jupyter/pyspark-notebook to your private registry
# China mirrors enabled by default for faster downloads.

FROM quay.io/jupyter/pyspark-notebook:latest

# Switch to root user to install system dependencies if needed
USER root

# Configure pip for China mirrors (Aliyun)
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.aliyun.com

# Switch back to jovyan user for pip installations (best practice for Jupyter images)
USER ${NB_UID}

# Install Python dependencies (Docker-specific)
COPY requirements.docker.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# Set working directory
WORKDIR /home/jovyan/work

# Expose Jupyter port
EXPOSE 8888
