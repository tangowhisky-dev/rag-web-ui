# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  default-libmysqlclient-dev \
  pkg-config \
  netcat-traditional \
  curl \
  ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# GPU support: build with --build-arg USE_GPU=true to install fastembed-gpu
# (onnxruntime-gpu) instead of fastembed (onnxruntime). The two are mutually
# exclusive — fastembed-gpu replaces fastembed and adds CUDAExecutionProvider.
# Requires NVIDIA Container Toolkit on the host and a GPU device reservation
# in docker-compose (see docker-compose.gpu.yml).
ARG USE_GPU=false
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$USE_GPU" = "true" ]; then \
        pip install -r requirements.txt && \
        pip uninstall -y fastembed onnxruntime && \
        pip install fastembed-gpu; \
    else \
        pip install -r requirements.txt; \
    fi

# Copy entrypoint script first
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Copy application files
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Set Python path and environment
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Run the application via entrypoint (migrations + readiness + uvicorn)
ENTRYPOINT ["./entrypoint.sh"]
