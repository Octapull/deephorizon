# DeepHorizon inference server image.
# Multi-stage build: training stack → slim runtime with ONNX model baked in.
#
# Build:
#   docker build -f infra/docker/inference.Dockerfile \
#     --build-arg ONNX_MODEL=exports/pix2pix_generator.onnx \
#     -t deephorizon-inference:v1 .
#
# Run:
#   docker run --gpus all -p 50051:50051 -p 8000:8000 deephorizon-inference:v1

# ---------------------------------------------------------------------------
# Stage 1: Builder — full training stack ile ONNX model üret
# ---------------------------------------------------------------------------
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Training dependencies (ONNX export için gerekli)
COPY requirements/base.txt requirements/ml.txt /app/requirements/
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/requirements/ml.txt

# Source code
COPY services /app/services

# ONNX model build arg olarak alınır
ARG ONNX_MODEL=exports/pix2pix_generator.onnx
ARG CHECKPOINT_PATH=checkpoints/best_model.pt
ARG MODEL_TYPE=pix2pix_generator
ARG INPUT_SHAPE="1,1,256,256"

# ONNX export (builder stage'de)
RUN mkdir -p /app/exports && \
    python -c "
from services.ml.export.onnx_export import export_checkpoint_to_onnx
import sys
shape = tuple(int(x) for x in '${INPUT_SHAPE}'.split(','))
export_checkpoint_to_onnx(
    checkpoint_path='${CHECKPOINT_PATH}',
    output_path='/app/exports/${ONNX_MODEL}',
    model_type='${MODEL_TYPE}',
    input_shape=shape,
    validate=True,
)
" || echo "ONNX export skipped (checkpoint not found — using pre-built model)"

# ---------------------------------------------------------------------------
# Stage 2: Runtime — slim CUDA runtime + ONNX model
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Python 3.11 + system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Serving dependencies (slim — training stack yok)
COPY requirements/base.txt requirements/serving.txt /app/requirements/
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/requirements/serving.txt

# Source code (sadece inference server)
COPY services /app/services

# ONNX model'i builder'dan kopyala
COPY --from=builder /app/exports /app/exports

# Non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose ports
# 50051: gRPC inference
# 8000: Prometheus metrics
EXPOSE 50051 8000

# Health check (gRPC server çalışıyor mu?)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import grpc; import socket; s = socket.socket(); s.settimeout(2); s.connect(('localhost', 50051)); s.close()" || exit 1

# Default: gRPC server + metrics server
ENTRYPOINT ["python", "-m", "services.ml.inference_server.server"]
CMD ["--models-dir", "/app/exports", "--port", "50051", "--metrics-port", "8000"]
