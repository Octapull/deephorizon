"""Prometheus metrics for the DeepHorizon inference server.

Exposes three core metrics for observability:

- ``inference_requests_total``: Counter — total inference requests,
  labeled by ``model_id`` and ``status`` (success/error).
- ``inference_latency_seconds``: Histogram — end-to-end inference
  latency (decode + inference + encode), labeled by ``model_id``.
- ``inference_batch_size``: Histogram — batch size distribution for
  ``EnhanceBatch`` RPCs.

The metrics are exposed via a Prometheus HTTP server on a configurable
port (default 8000). The server runs in a daemon thread alongside the
gRPC server.

Usage:
    from services.ml.inference_server.metrics import (
        start_metrics_server,
        record_inference,
        record_batch_size,
    )

    start_metrics_server(port=8000)
    record_inference(model_id="pix2pix-v1", status="success", latency=0.123)
    record_batch_size(model_id="pix2pix-v1", batch_size=4)

Reference:
    Prometheus Python client: https://github.com/prometheus/client_python
"""

from __future__ import annotations

import logging
import threading
from typing import Literal

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    start_http_server,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global registry — tek bir registry kullanıyoruz (test'lerde override edilebilir)
# ---------------------------------------------------------------------------
_REGISTRY = CollectorRegistry()

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------
INFERENCE_REQUESTS_TOTAL = Counter(
    "inference_requests_total",
    "Total number of inference requests served.",
    labelnames=("model_id", "status"),
    registry=_REGISTRY,
)

INFERENCE_LATENCY_SECONDS = Histogram(
    "inference_latency_seconds",
    "End-to-end inference latency in seconds (decode + inference + encode).",
    labelnames=("model_id",),
    registry=_REGISTRY,
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

INFERENCE_BATCH_SIZE = Histogram(
    "inference_batch_size",
    "Distribution of batch sizes for EnhanceBatch RPCs.",
    labelnames=("model_id",),
    registry=_REGISTRY,
    buckets=(1, 2, 4, 8, 16, 32, 64),
)

INFERENCE_MODEL_LOAD_DURATION = Histogram(
    "inference_model_load_duration_seconds",
    "Time to load an ONNX model into memory (lazy load on first request).",
    labelnames=("model_id",),
    registry=_REGISTRY,
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
_metrics_server_started = False
_metrics_server_lock = threading.Lock()


def start_metrics_server(port: int = 8000, addr: str = "0.0.0.0") -> None:
    """Prometheus HTTP server'ı başlat (daemon thread).

    Args:
        port: HTTP port (default 8000).
        addr: Bind address (default 0.0.0.0 — tüm interface'lerden erişilebilir).
    """
    global _metrics_server_started

    with _metrics_server_lock:
        if _metrics_server_started:
            logger.debug("Metrics server already started on port %d", port)
            return

        try:
            start_http_server(port=port, addr=addr, registry=_REGISTRY)
            _metrics_server_started = True
            logger.info(
                "Prometheus metrics server started on %s:%d/metrics", addr, port
            )
        except OSError as exc:
            logger.warning(
                "Could not start metrics server on %s:%d: %s. "
                "Metrics will not be exposed.",
                addr,
                port,
                exc,
            )


def is_metrics_server_started() -> bool:
    """Metrics server başlatıldı mı?"""
    return _metrics_server_started


# ---------------------------------------------------------------------------
# Recording helpers
# ---------------------------------------------------------------------------
Status = Literal["success", "error", "not_found", "invalid_argument"]


def record_inference(
    model_id: str,
    status: Status,
    latency_seconds: float,
) -> None:
    """Bir inference isteğini kaydet.

    Args:
        model_id: Model identifier (e.g. ``"pix2pix-v1"``).
        status: İstek durumu — ``"success"``, ``"error"``, ``"not_found"``,
            ``"invalid_argument"``.
        latency_seconds: End-to-end latency (saniye).
    """
    INFERENCE_REQUESTS_TOTAL.labels(model_id=model_id, status=status).inc()
    INFERENCE_LATENCY_SECONDS.labels(model_id=model_id).observe(latency_seconds)


def record_batch_size(model_id: str, batch_size: int) -> None:
    """Batch size dağılımını kaydet.

    Args:
        model_id: Model identifier.
        batch_size: Batch içindeki görüntü sayısı.
    """
    INFERENCE_BATCH_SIZE.labels(model_id=model_id).observe(batch_size)


def record_model_load(model_id: str, duration_seconds: float) -> None:
    """Model yükleme süresini kaydet.

    Args:
        model_id: Model identifier.
        duration_seconds: Yükleme süresi (saniye).
    """
    INFERENCE_MODEL_LOAD_DURATION.labels(model_id=model_id).observe(duration_seconds)


def get_registry() -> CollectorRegistry:
    """Test'ler için registry erişimi."""
    return _REGISTRY
