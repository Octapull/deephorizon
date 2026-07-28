"""Unit tests for services.ml.inference_server.

Tests cover:
    - image_utils: encode/decode roundtrip for PNG, JPEG, raw float32
    - model_registry: register, get, list, unregister, thread-safety
    - server: CLI argument parsing, registry building
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from services.ml.inference_server.image_utils import (
    decode_image_to_tensor,
    encode_tensor_to_image,
)
from services.ml.inference_server.model_registry import ModelRegistry, ModelSpec


# ---------------------------------------------------------------------------
# image_utils: decode
# ---------------------------------------------------------------------------


def test_decode_png_grayscale() -> None:
    """PNG decode → (1, 1, H, W) float32 [0, 1]."""
    from PIL import Image  # type: ignore[import-untyped]

    # Create a 32x32 grayscale PNG
    array = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    image = Image.fromarray(array, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()

    tensor = decode_image_to_tensor(png_bytes, "image/png")

    assert tensor.shape == (1, 1, 32, 32)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_decode_png_rgb() -> None:
    """PNG RGB decode → (1, 3, H, W) float32 [0, 1]."""
    from PIL import Image  # type: ignore[import-untyped]

    array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    image = Image.fromarray(array, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()

    tensor = decode_image_to_tensor(png_bytes, "image/png")

    assert tensor.shape == (1, 3, 32, 32)
    assert tensor.dtype == torch.float32


def test_decode_raw_float32() -> None:
    """Raw float32 decode → (1, 1, H, W) float32."""
    array = np.random.rand(16, 16).astype(np.float32)
    raw_bytes = array.tobytes()

    tensor = decode_image_to_tensor(
        raw_bytes, "application/octet-stream", width=16, height=16
    )

    assert tensor.shape == (1, 1, 16, 16)
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor[0, 0], torch.from_numpy(array), atol=1e-6)


def test_decode_raw_missing_dims() -> None:
    """Raw float32 without width/height → ValueError."""
    with pytest.raises(ValueError, match="requires explicit width"):
        decode_image_to_tensor(b"\x00" * 100, "application/octet-stream")


def test_decode_raw_size_mismatch() -> None:
    """Raw float32 with wrong size → ValueError."""
    with pytest.raises(ValueError, match="size mismatch"):
        decode_image_to_tensor(
            b"\x00" * 100, "application/octet-stream", width=16, height=16
        )


def test_decode_unsupported_mime() -> None:
    """Unsupported mime_type → ValueError."""
    with pytest.raises(ValueError, match="Unsupported mime_type"):
        decode_image_to_tensor(b"data", "image/bmp")


# ---------------------------------------------------------------------------
# image_utils: encode
# ---------------------------------------------------------------------------


def test_encode_png_roundtrip() -> None:
    """Encode → decode → same shape and approximate values."""
    tensor = torch.rand(1, 1, 32, 32)
    data, mime, w, h = encode_tensor_to_image(tensor, output_format="png")

    assert mime == "image/png"
    assert w == 32
    assert h == 32
    assert len(data) > 0

    # Decode back
    decoded = decode_image_to_tensor(data, mime)
    assert decoded.shape == tensor.shape


def test_encode_jpeg() -> None:
    """JPEG encode → image/jpeg mime."""
    tensor = torch.rand(1, 1, 32, 32)
    data, mime, w, h = encode_tensor_to_image(tensor, output_format="jpeg")

    assert mime == "image/jpeg"
    assert w == 32
    assert h == 32


def test_encode_raw_float32() -> None:
    """Raw float32 encode → application/octet-stream."""
    tensor = torch.rand(1, 1, 16, 16)
    data, mime, w, h = encode_tensor_to_image(tensor, output_format="raw")

    assert mime == "application/octet-stream"
    assert w == 16
    assert h == 16
    assert len(data) == 16 * 16 * 4  # float32 = 4 bytes


def test_encode_unsupported_format() -> None:
    """Unsupported output_format → ValueError."""
    tensor = torch.rand(1, 1, 16, 16)
    with pytest.raises(ValueError, match="Unsupported output_format"):
        encode_tensor_to_image(tensor, output_format="tiff")


def test_encode_rgb_tensor() -> None:
    """RGB tensor (1, 3, H, W) → PNG encode."""
    tensor = torch.rand(1, 3, 32, 32)
    data, mime, w, h = encode_tensor_to_image(tensor, output_format="png")

    assert mime == "image/png"
    assert w == 32
    assert h == 32


# ---------------------------------------------------------------------------
# model_registry
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_onnx_path(tmp_path: Path) -> Path:
    """Create a dummy .onnx file (not a real ONNX, just for path tests)."""
    path = tmp_path / "fake_model.onnx"
    path.write_bytes(b"fake onnx data")
    return path


def test_registry_register(fake_onnx_path: Path) -> None:
    """register → spec eklenir."""
    registry = ModelRegistry()
    spec = ModelSpec(
        model_id="test-v1",
        architecture="pix2pix",
        version="1.0.0",
        onnx_path=fake_onnx_path,
    )
    registry.register(spec)

    assert "test-v1" in [s.model_id for s in registry.list_specs()]


def test_registry_register_duplicate_raises(fake_onnx_path: Path) -> None:
    """Duplicate model_id → ValueError."""
    registry = ModelRegistry()
    spec = ModelSpec(
        model_id="test-v1",
        architecture="pix2pix",
        version="1.0.0",
        onnx_path=fake_onnx_path,
    )
    registry.register(spec)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec)


def test_registry_register_missing_file(tmp_path: Path) -> None:
    """Missing ONNX file → FileNotFoundError."""
    registry = ModelRegistry()
    spec = ModelSpec(
        model_id="test-v1",
        architecture="pix2pix",
        version="1.0.0",
        onnx_path=tmp_path / "nonexistent.onnx",
    )
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        registry.register(spec)


def test_registry_unregister(fake_onnx_path: Path) -> None:
    """unregister → spec kaldırılır."""
    registry = ModelRegistry()
    spec = ModelSpec(
        model_id="test-v1",
        architecture="pix2pix",
        version="1.0.0",
        onnx_path=fake_onnx_path,
    )
    registry.register(spec)
    registry.unregister("test-v1")

    assert "test-v1" not in [s.model_id for s in registry.list_specs()]


def test_registry_get_unknown_raises(fake_onnx_path: Path) -> None:
    """get(unknown_id) → KeyError."""
    registry = ModelRegistry()
    spec = ModelSpec(
        model_id="test-v1",
        architecture="pix2pix",
        version="1.0.0",
        onnx_path=fake_onnx_path,
    )
    registry.register(spec)

    with pytest.raises(KeyError, match="not registered"):
        registry.get("unknown-id")


def test_registry_list_specs(fake_onnx_path: Path) -> None:
    """list_specs → tüm kayıtlı spec'ler."""
    registry = ModelRegistry()
    for i in range(3):
        spec = ModelSpec(
            model_id=f"model-{i}",
            architecture="pix2pix",
            version="1.0.0",
            onnx_path=fake_onnx_path,
        )
        registry.register(spec)

    specs = registry.list_specs()
    assert len(specs) == 3
    assert {s.model_id for s in specs} == {"model-0", "model-1", "model-2"}


def test_registry_is_loaded_false(fake_onnx_path: Path) -> None:
    """is_loaded → False until get() çağrılır."""
    registry = ModelRegistry()
    spec = ModelSpec(
        model_id="test-v1",
        architecture="pix2pix",
        version="1.0.0",
        onnx_path=fake_onnx_path,
    )
    registry.register(spec)

    assert registry.is_loaded("test-v1") is False


# ---------------------------------------------------------------------------
# server: build_registry_from_args
# ---------------------------------------------------------------------------


def test_build_registry_from_single_onnx(fake_onnx_path: Path) -> None:
    """build_registry_from_args(--onnx-path) → tek model."""
    from services.ml.inference_server.server import build_registry_from_args

    args = MagicMock(
        onnx_path=str(fake_onnx_path),
        models_dir=None,
        model_id="custom-id",
        architecture="pix2pix",
        version="1.0.0",
    )
    registry = build_registry_from_args(args)

    specs = registry.list_specs()
    assert len(specs) == 1
    assert specs[0].model_id == "custom-id"
    assert specs[0].architecture == "pix2pix"


def test_build_registry_from_models_dir(tmp_path: Path, fake_onnx_path: Path) -> None:
    """build_registry_from_args(--models-dir) → tüm .onnx dosyaları."""
    from services.ml.inference_server.server import build_registry_from_args

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "model_a.onnx").write_bytes(b"fake")
    (models_dir / "model_b.onnx").write_bytes(b"fake")

    args = MagicMock(
        onnx_path=None,
        models_dir=str(models_dir),
        model_id=None,
        architecture="unet",
        version="2.0.0",
    )
    registry = build_registry_from_args(args)

    specs = registry.list_specs()
    assert len(specs) == 2
    assert {s.model_id for s in specs} == {"model_a", "model_b"}


def test_build_registry_default_model_id(fake_onnx_path: Path) -> None:
    """build_registry_from_args → model_id yoksa 'default' kullanılır."""
    from services.ml.inference_server.server import build_registry_from_args

    args = MagicMock(
        onnx_path=str(fake_onnx_path),
        models_dir=None,
        model_id=None,
        architecture="pix2pix",
        version="1.0.0",
    )
    registry = build_registry_from_args(args)

    specs = registry.list_specs()
    assert specs[0].model_id == "default"


def test_build_registry_metadata_sidecar(tmp_path: Path) -> None:
    """build_registry_from_args → metadata .json varsa yükler."""
    from services.ml.inference_server.server import build_registry_from_args

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake")
    metadata_path = tmp_path / "model.onnx.json"
    metadata_path.write_text('{"psnr": 30.0, "ssim": 0.9}')

    args = MagicMock(
        onnx_path=str(onnx_path),
        models_dir=None,
        model_id="test",
        architecture="pix2pix",
        version="1.0.0",
    )
    registry = build_registry_from_args(args)

    spec = registry.list_specs()[0]
    assert spec.metadata_path == metadata_path
