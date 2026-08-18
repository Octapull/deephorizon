# DeepHorizon remote GPU training image.
# Built on the GPU server and pushed to the single-node MicroK8s registry.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install the reproducible training dependency set before copying source code
# so normal code edits can reuse the expensive dependency layer.
COPY requirements/base.txt requirements/ml.txt /app/requirements/
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/requirements/ml.txt

COPY services /app/services

# Kubernetes Job args are Hydra overrides appended to this entrypoint.
ENTRYPOINT ["python", "-m", "services.ml.training.train"]
