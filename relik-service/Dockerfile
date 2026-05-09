# syntax=docker/dockerfile:1
# Custom ReLiK service image — built for arm64 (Apple Silicon) and amd64.
# The official sapienzanlp/relik:latest is amd64-only and performs poorly under
# Docker on MacOS. This image is platform-native and installs PyTorch CPU wheels
# from the official index so pip picks the correct ABI for the host arch.

FROM python:3.11-slim

WORKDIR /app

# System deps: git is needed by some HuggingFace model downloads; curl for healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first (smaller, arch-correct wheels).
# relik[serve] pins uvicorn==0.23.2 — don't add a separate uvicorn pin.
# numpy<2 required: relik transitive deps were built against numpy 1.x.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install \
        "torch==2.3.1" \
        "torchvision==0.18.1" \
        "torchaudio==2.3.1" \
        "relik[serve]>=0.4.0" \
        "numpy<2.0.0" \
        --extra-index-url https://download.pytorch.org/whl/cpu

COPY serve.py .

EXPOSE 8000

# Model is downloaded on first startup from HuggingFace Hub.
# Mount ./assets/relik-hf-cache:/root/.cache/huggingface/hub to persist it.
ENV RELIK_MODEL=relik-ie/relik-cie-small
ENV RELIK_DEVICE=cpu

# Healthcheck — service is ready once the root endpoint returns 200.
HEALTHCHECK --interval=15s --timeout=10s --start-period=120s --retries=20 \
    CMD curl -sf http://localhost:8000/ || exit 1

CMD ["python", "serve.py"]
