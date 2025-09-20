# Lightweight Python image
FROM python:3.11-slim

# System deps: ffmpeg for format conversions, git for installing chatterbox from GitHub
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       git \
       build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install (kept minimal)
COPY requirements.txt /app/requirements.txt
# Install deps in a stable order to satisfy packages that import numpy at build time
RUN pip install --no-cache-dir "numpy<1.26" \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app

ENV PYTHONPATH="/app/src"

# Create assets directories (persisted if mounted)
RUN mkdir -p /app/assets/inputs /app/assets/outputs

# Default command runs the interactive CLI
ENTRYPOINT ["python", "-m", "tts_cli.cli"]
