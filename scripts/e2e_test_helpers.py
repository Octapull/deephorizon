"""Shared helpers for end-to-end integration tests.

Provides utilities for:
- Generating synthetic test images (no external data needed)
- Encoding/decoding images for gRPC transport
- Spinning up the Python inference server in-process
- Sending gRPC requests and validating responses
- Cleanup of temporary files and processes

These helpers are used by ``e2e_integration_test.py`` and can be
reused for future integration tests.
"""
from __future__ import annotations

import io
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic image generation
# ---------------------------------------------------------------------------


def generate_synthetic_image(
    width: int = 64,
    height: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic black-hole-like image.

    Creates a 2D array with:
    - Central bright spot (accretion disk)
    - Surrounding ring (photon ring)
    - Gaussian noise (simulating telescope noise)

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        seed: Random seed for reproducibility.

    Returns:
        Float32 array of shape ``(height, width)`` in ``[0, 1]``.
    """
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width), dtype=np.float32)

    # Central bright spot
    cy, cx = height / 2.0, width / 2.0
    y_coords, x_coords = np.ogrid[:height, :width]
    r = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)

    # Accretion disk (Gaussian)
    disk_sigma = min(width, height) / 8.0
    image += np.exp(-(r ** 2) / (2.0 * disk_sigma ** 2))

    # Photon ring (narrow ring at radius ~ min_dim/4)
    ring_radius = min(width, height) / 4.0
    ring_width = 2.0
    image += 0.5 * np.exp(-((r - ring_radius) ** 2) / (2.0 * ring_width ** 2))

    # Noise
    image += rng.normal(0, 0.05, image.shape).astype(np.float32)

    # Clip to [0, 1]
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def generate_synthetic_batch(
    batch_size: int = 4,
    width: int = 64,
    height: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """Generate a batch of synthetic images.

    Returns:
        Float32 array of shape ``(batch_size, height, width)`` in ``[0, 1]``.
    """
    images = []
    for i in range(batch_size):
        images.append(generate_synthetic_image(width=width, height=height, seed=seed + i))
    return np.stack(images, axis=0)


# ---------------------------------------------------------------------------
# Image encoding for gRPC transport
# ---------------------------------------------------------------------------


def numpy_to_png_bytes(image: np.ndarray) -> bytes:
    """Encode a 2D numpy array as PNG bytes.

    Args:
        image: 2D float32 array in ``[0, 1]``.

    Returns:
        PNG-encoded bytes.
    """
    from PIL import Image  # type: ignore[import-untyped]

    array_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil_image = Image.fromarray(array_uint8, mode="L")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def png_bytes_to_numpy(data: bytes) -> np.ndarray:
    """Decode PNG bytes to a 2D numpy array.

    Returns:
        Float32 array in ``[0, 1]``.
    """
    from PIL import Image  # type: ignore[import-untyped]

    pil_image = Image.open(io.BytesIO(data))
    return np.asarray(pil_image, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Port management
# ---------------------------------------------------------------------------


def find_free_port() -> int:
    """Find an available TCP port on localhost.

    Returns:
        A free port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_port(host: str, port: int, timeout: float = 30.0, poll_interval: float = 0.5) -> bool:
    """Wait until a TCP port is accepting connections.

    Args:
        host: Hostname to check.
        port: Port number to check.
        timeout: Max time to wait in seconds.
        poll_interval: Time between connection attempts.

    Returns:
        True if port became available, False if timeout reached.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(poll_interval)
    return False


# ---------------------------------------------------------------------------
# Inference server lifecycle
# ---------------------------------------------------------------------------


@contextmanager
def inference_server(
    onnx_path: Path | str,
    model_id: str = "test-model",
    port: int | None = None,
    startup_timeout: float = 30.0,
    log_file: Path | str | None = None,
) -> Iterator[dict]:
    """Start the Python inference server as a subprocess.

    Args:
        onnx_path: Path to the ONNX model file.
        model_id: Model identifier to register.
        port: TCP port (auto-detected if None).
        startup_timeout: Max time to wait for server startup.
        log_file: Optional path to write server logs.

    Yields:
        Dict with keys:
            - ``process``: subprocess.Popen handle
            - ``port``: actual port used
            - ``model_id``: model identifier
            - ``onnx_path``: path to ONNX model

    Example:
        >>> with inference_server("model.onnx") as ctx:
        ...     # Server is running on ctx["port"]
        ...     # Send gRPC requests...
        ... # Server is automatically stopped
    """
    if port is None:
        port = find_free_port()

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    cmd = [
        sys.executable,
        "-m",
        "services.ml.inference_server.server",
        "--onnx-path",
        str(onnx_path),
        "--model-id",
        model_id,
        "--port",
        str(port),
        "--log-level",
        "WARNING",
    ]

    log_handle = None
    if log_file is not None:
        log_handle = open(log_file, "w", encoding="utf-8")

    process = subprocess.Popen(
        cmd,
        stdout=log_handle or subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        cwd=Path(__file__).resolve().parent.parent,
    )

    try:
        # Wait for server to start accepting connections
        if not wait_for_port("127.0.0.1", port, timeout=startup_timeout):
            process.terminate()
            process.wait(timeout=5)
            raise RuntimeError(
                f"Inference server failed to start on port {port} within {startup_timeout}s"
            )

        yield {
            "process": process,
            "port": port,
            "model_id": model_id,
            "onnx_path": str(onnx_path),
        }
    finally:
        # Cleanup
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if log_handle is not None:
            log_handle.close()


# ---------------------------------------------------------------------------
# gRPC client helpers
# ---------------------------------------------------------------------------


def create_grpc_channel(host: str = "127.0.0.1", port: int = 50051):
    """Create an insecure gRPC channel.

    Returns:
        grpc.Channel object.
    """
    import grpc  # type: ignore[import-untyped]

    return grpc.insecure_channel(f"{host}:{port}")


def get_inference_stub(channel):
    """Get the InferenceService stub from a gRPC channel.

    Returns:
        InferenceServiceStub instance.
    """
    from services.ml.inference_server.pb import inference_pb2_grpc  # type: ignore[import-untyped]

    return inference_pb2_grpc.InferenceServiceStub(channel)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_enhance_response(
    response,
    expected_status: int = 3,  # JOB_STATUS_COMPLETED
    expected_mime_prefix: str = "image/",
    min_output_size: int = 100,
) -> dict:
    """Validate an EnhanceResponse and return a summary dict.

    Args:
        response: EnhanceResponse proto message.
        expected_status: Expected JobStatus enum value.
        expected_mime_prefix: Expected MIME type prefix.
        min_output_size: Minimum output image size in bytes.

    Returns:
        Dict with validation results:
            - ``status_ok``: bool
            - ``mime_ok``: bool
            - ``size_ok``: bool
            - ``has_metrics``: bool
            - ``inference_time_ms``: int
            - ``output_shape``: tuple or None
    """
    from services.ml.inference_server.pb import inference_pb2  # type: ignore[import-untyped]

    status_ok = response.status == expected_status
    mime_ok = response.image.mime_type.startswith(expected_mime_prefix)
    size_ok = len(response.image.data) >= min_output_size
    has_metrics = response.metrics.inference_time_ms > 0

    output_shape = None
    if response.image.width > 0 and response.image.height > 0:
        output_shape = (response.image.width, response.image.height)

    return {
        "status_ok": status_ok,
        "mime_ok": mime_ok,
        "size_ok": size_ok,
        "has_metrics": has_metrics,
        "inference_time_ms": response.metrics.inference_time_ms,
        "output_shape": output_shape,
        "model_id": response.model_id,
    }


def assert_response_valid(result: dict, context: str = "") -> None:
    """Assert that a validation result indicates a successful response.

    Raises:
        AssertionError: If any check failed.
    """
    prefix = f"[{context}] " if context else ""
    assert result["status_ok"], f"{prefix}status check failed: {result}"
    assert result["mime_ok"], f"{prefix}mime check failed: {result}"
    assert result["size_ok"], f"{prefix}size check failed: {result}"
    assert result["has_metrics"], f"{prefix}metrics check failed: {result}"
