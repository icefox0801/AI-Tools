# Use the official Jupyter PySpark notebook as base image
FROM quay.io/jupyter/pyspark-notebook:latest

# Switch to root user to install system dependencies if needed
USER root

# Switch back to jovyan user for pip installations (best practice for Jupyter images)
USER ${NB_UID}

# Install Python dependencies (Docker-specific)
COPY requirements.docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set working directory
WORKDIR /home/jovyan/work

# Expose Jupyter port
EXPOSE 8888
