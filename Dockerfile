# OpenGrid Docker Image
# Production-ready container for OpenGrid Power Systems Analysis Platform
#
# Author: Nik Jois (nikjois@llamasearch.ai)
# License: MIT

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create application user
RUN groupadd -r opengrid && useradd -r -g opengrid opengrid

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    git \
    curl \
    libffi-dev \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/results

# Set permissions
RUN chown -R opengrid:opengrid /app

# Switch to non-root user
USER opengrid

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "main.py"]

# Add labels
LABEL \
    maintainer="Nik Jois <nikjois@llamasearch.ai>" \
    description="OpenGrid Power Systems Analysis Platform" \
    version="0.2.0" \
    org.opencontainers.image.source="https://github.com/llamasearchai/OpenGrid" \
    org.opencontainers.image.documentation="https://github.com/llamasearchai/OpenGrid/blob/main/README.md" \
    org.opencontainers.image.licenses="MIT" 