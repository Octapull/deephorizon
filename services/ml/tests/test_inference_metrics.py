"""Unit tests for services.ml.inference_server.metrics."""

from __future__ import annotations

from prometheus_client import generate_latest

from services.ml.inference_server.metrics import (
    INFERENCE_BATCH_SIZE,
    INFERENCE_LATENCY_SECONDS,
    INFERENCE_REQUESTS_TOTAL,
    get_registry,
    record_batch_size,
    record_inference,
)


def test_record_inference_success() -> None:
    """record_inference(success) → counter artar, latency kaydedilir."""
    # Başlangıç değerini al
    registry = get_registry()
    output_before = generate_latest(registry).decode("utf-8")

    record_inference(model_id="test-model", status="success", latency_seconds=0.123)

    output_after = generate_latest(registry).decode("utf-8")

    # Counter artmış olmalı
    assert "inference_requests_total" in output_after
    assert 'model_id="test-model"' in output_after
    assert 'status="success"' in output_after
    # Histogram observation kaydedilmiş olmalı
    assert "inference_latency_seconds" in output_after


def test_record_inference_error() -> None:
    """record_inference(error) → error label ile counter artar."""
    record_inference(model_id="test-model", status="error", latency_seconds=0.5)

    registry = get_registry()
    output = generate_latest(registry).decode("utf-8")

    assert 'status="error"' in output


def test_record_inference_not_found() -> None:
    """record_inference(not_found) → not_found label ile counter artar."""
    record_inference(model_id="test-model", status="not_found", latency_seconds=0.01)

    registry = get_registry()
    output = generate_latest(registry).decode("utf-8")

    assert 'status="not_found"' in output


def test_record_batch_size() -> None:
    """record_batch_size → batch size histogram'a kaydedilir."""
    record_batch_size(model_id="test-model", batch_size=8)

    registry = get_registry()
    output = generate_latest(registry).decode("utf-8")

    assert "inference_batch_size" in output
    assert 'model_id="test-model"' in output


def test_metrics_registry_isolated() -> None:
    """get_registry() → aynı registry instance'ını döner."""
    registry1 = get_registry()
    registry2 = get_registry()

    assert registry1 is registry2


def test_metrics_have_correct_labels() -> None:
    """Metric'ler doğru label'lara sahip."""
    # Counter
    assert "model_id" in INFERENCE_REQUESTS_TOTAL._labelnames
    assert "status" in INFERENCE_REQUESTS_TOTAL._labelnames

    # Histogram
    assert "model_id" in INFERENCE_LATENCY_SECONDS._labelnames
    assert "model_id" in INFERENCE_BATCH_SIZE._labelnames


def test_metrics_output_format() -> None:
    """generate_latest() → Prometheus exposition format."""
    record_inference(model_id="format-test", status="success", latency_seconds=0.1)

    registry = get_registry()
    output = generate_latest(registry).decode("utf-8")

    # Prometheus exposition format kontrolü
    assert "# HELP inference_requests_total" in output
    assert "# TYPE inference_requests_total counter" in output
    assert "# HELP inference_latency_seconds" in output
    assert "# TYPE inference_latency_seconds histogram" in output
