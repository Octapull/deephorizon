"""Unit tests for services.ml.export.onnx_export."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from services.ml.export.onnx_export import (
    ExportMetadata,
    export_checkpoint_to_onnx,
    export_to_onnx,
    load_metadata,
    load_model_from_checkpoint,
    save_metadata,
)
from services.ml.models.pix2pix import Pix2PixGenerator
from services.ml.models.unet import UNet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def unet_checkpoint(tmp_path: Path) -> Path:
    """Create a dummy U-Net checkpoint."""
    model = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "unet_ckpt.pt"

    torch.save(
        {
            "epoch": 5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": 0.3,
            "val_loss": 0.4,
            "psnr": 28.5,
            "ssim": 0.85,
        },
        path,
    )
    return path


@pytest.fixture
def pix2pix_checkpoint(tmp_path: Path) -> Path:
    """Create a dummy Pix2Pix GAN checkpoint."""
    generator = Pix2PixGenerator(in_channels=1, out_channels=1, dropout=0.0)
    discriminator = torch.nn.Module()  # placeholder
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=5e-4)
    path = tmp_path / "gan_ckpt.pt"

    torch.save(
        {
            "epoch": 10,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "g_optimizer_state_dict": g_optimizer.state_dict(),
            "d_optimizer_state_dict": d_optimizer.state_dict(),
            "train_loss": 0.2,
            "val_loss": 0.3,
            "psnr": 30.0,
            "ssim": 0.90,
        },
        path,
    )
    return path


# ---------------------------------------------------------------------------
# load_model_from_checkpoint
# ---------------------------------------------------------------------------


def test_load_unet_from_checkpoint(unet_checkpoint: Path) -> None:
    """load_model_from_checkpoint('unet') → UNet in eval mode."""
    model, checkpoint = load_model_from_checkpoint(
        unet_checkpoint, model_type="unet"
    )

    assert isinstance(model, UNet)
    assert not model.training  # eval mode
    assert checkpoint["epoch"] == 5
    assert checkpoint["psnr"] == 28.5


def test_load_pix2pix_from_checkpoint(pix2pix_checkpoint: Path) -> None:
    """load_model_from_checkpoint('pix2pix_generator') → Pix2PixGenerator."""
    model, checkpoint = load_model_from_checkpoint(
        pix2pix_checkpoint, model_type="pix2pix_generator"
    )

    assert isinstance(model, Pix2PixGenerator)
    assert not model.training
    assert checkpoint["epoch"] == 10
    assert checkpoint["psnr"] == 30.0


def test_load_unknown_model_type(unet_checkpoint: Path) -> None:
    """Unknown model_type → ValueError."""
    with pytest.raises(ValueError, match="Unknown model_type"):
        load_model_from_checkpoint(unet_checkpoint, model_type="vae")


def test_load_missing_checkpoint(tmp_path: Path) -> None:
    """Non-existent checkpoint → FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_model_from_checkpoint(
            tmp_path / "nonexistent.pt", model_type="unet"
        )


# ---------------------------------------------------------------------------
# export_to_onnx
# ---------------------------------------------------------------------------


def test_export_unet_to_onnx(tmp_path: Path, unet_checkpoint: Path) -> None:
    """export_to_onnx(UNet) → .onnx file oluşturur."""
    model, _ = load_model_from_checkpoint(unet_checkpoint, model_type="unet")
    output_path = tmp_path / "unet.onnx"

    result_path = export_to_onnx(
        model=model,
        output_path=output_path,
        input_shape=(1, 1, 64, 64),
        validate=False,  # onnxruntime may not be installed
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_export_pix2pix_to_onnx(tmp_path: Path, pix2pix_checkpoint: Path) -> None:
    """export_to_onnx(Pix2PixGenerator) → .onnx file oluşturur."""
    model, _ = load_model_from_checkpoint(
        pix2pix_checkpoint, model_type="pix2pix_generator"
    )
    output_path = tmp_path / "pix2pix.onnx"

    result_path = export_to_onnx(
        model=model,
        output_path=output_path,
        input_shape=(1, 1, 64, 64),
        validate=False,
    )

    assert result_path == output_path
    assert output_path.exists()


def test_export_creates_parent_dir(tmp_path: Path, unet_checkpoint: Path) -> None:
    """export_to_onnx → üst dizini otomatik oluşturur."""
    model, _ = load_model_from_checkpoint(unet_checkpoint, model_type="unet")
    output_path = tmp_path / "nested" / "dir" / "model.onnx"

    export_to_onnx(
        model=model,
        output_path=output_path,
        input_shape=(1, 1, 32, 32),
        validate=False,
    )

    assert output_path.exists()


def test_export_custom_opset(tmp_path: Path, unet_checkpoint: Path) -> None:
    """export_to_onnx(onnx_opset=13) → farklı opset ile export."""
    model, _ = load_model_from_checkpoint(unet_checkpoint, model_type="unet")
    output_path = tmp_path / "model_op13.onnx"

    export_to_onnx(
        model=model,
        output_path=output_path,
        input_shape=(1, 1, 32, 32),
        onnx_opset=13,
        validate=False,
    )

    assert output_path.exists()


def test_export_no_dynamic_axes(tmp_path: Path, unet_checkpoint: Path) -> None:
    """export_to_onnx(dynamic_axes=False) → sabit batch size."""
    model, _ = load_model_from_checkpoint(unet_checkpoint, model_type="unet")
    output_path = tmp_path / "model_static.onnx"

    export_to_onnx(
        model=model,
        output_path=output_path,
        input_shape=(2, 1, 32, 32),
        dynamic_axes=False,
        validate=False,
    )

    assert output_path.exists()


# ---------------------------------------------------------------------------
# Metadata sidecar
# ---------------------------------------------------------------------------


def test_save_and_load_metadata(tmp_path: Path) -> None:
    """save_metadata + load_metadata → roundtrip."""
    metadata = ExportMetadata(
        model_name="test_model",
        model_type="pix2pix_generator",
        onnx_opset=17,
        input_shape=(1, 1, 256, 256),
        output_shape=(1, 1, 256, 256),
        input_dtype="float32",
        output_dtype="float32",
        checkpoint_path="/path/to/ckpt.pt",
        checkpoint_epoch=10,
        val_loss=0.3,
        psnr=30.0,
        ssim=0.9,
        export_timestamp="2026-07-28T12:00:00+00:00",
    )

    onnx_path = tmp_path / "model.onnx"
    onnx_path.touch()  # placeholder

    metadata_path = save_metadata(metadata, onnx_path)
    assert metadata_path.exists()
    assert metadata_path == onnx_path.with_suffix(".onnx.json")

    loaded = load_metadata(metadata_path)
    assert loaded.model_name == "test_model"
    assert loaded.model_type == "pix2pix_generator"
    assert loaded.onnx_opset == 17
    assert loaded.input_shape == (1, 1, 256, 256)
    assert loaded.psnr == 30.0


def test_metadata_json_format(tmp_path: Path) -> None:
    """Metadata JSON dosyası geçerli JSON."""
    metadata = ExportMetadata(
        model_name="test",
        model_type="unet",
        onnx_opset=17,
        input_shape=(1, 1, 64, 64),
        output_shape=(1, 1, 64, 64),
        input_dtype="float32",
        output_dtype="float32",
        checkpoint_path="/ckpt.pt",
        checkpoint_epoch=1,
        val_loss=0.5,
        psnr=25.0,
        ssim=0.8,
        export_timestamp="2026-01-01T00:00:00+00:00",
    )

    onnx_path = tmp_path / "model.onnx"
    onnx_path.touch()
    metadata_path = save_metadata(metadata, onnx_path)

    # JSON dosyası parse edilebilmeli
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["model_name"] == "test"
    assert data["model_type"] == "unet"
    assert data["input_shape"] == [1, 1, 64, 64]


# ---------------------------------------------------------------------------
# export_checkpoint_to_onnx (end-to-end)
# ---------------------------------------------------------------------------


def test_export_checkpoint_to_onnx_unet(tmp_path: Path, unet_checkpoint: Path) -> None:
    """export_checkpoint_to_onnx → ONNX + metadata oluşturur."""
    output_path = tmp_path / "unet_export.onnx"

    onnx_path, metadata_path = export_checkpoint_to_onnx(
        checkpoint_path=unet_checkpoint,
        output_path=output_path,
        model_type="unet",
        input_shape=(1, 1, 64, 64),
        validate=False,
    )

    assert onnx_path == output_path
    assert onnx_path.exists()
    assert metadata_path.exists()
    assert metadata_path == output_path.with_suffix(".onnx.json")

    # Metadata doğrulama
    metadata = load_metadata(metadata_path)
    assert metadata.model_type == "unet"
    assert metadata.checkpoint_epoch == 5
    assert metadata.psnr == 28.5


def test_export_checkpoint_to_onnx_pix2pix(
    tmp_path: Path, pix2pix_checkpoint: Path
) -> None:
    """export_checkpoint_to_onnx(pix2pix) → ONNX + metadata."""
    output_path = tmp_path / "pix2pix_export.onnx"

    onnx_path, metadata_path = export_checkpoint_to_onnx(
        checkpoint_path=pix2pix_checkpoint,
        output_path=output_path,
        model_type="pix2pix_generator",
        input_shape=(1, 1, 64, 64),
        validate=False,
    )

    assert onnx_path.exists()
    assert metadata_path.exists()

    metadata = load_metadata(metadata_path)
    assert metadata.model_type == "pix2pix_generator"
    assert metadata.checkpoint_epoch == 10
    assert metadata.psnr == 30.0


def test_export_checkpoint_missing_file(tmp_path: Path) -> None:
    """Non-existent checkpoint → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        export_checkpoint_to_onnx(
            checkpoint_path=tmp_path / "missing.pt",
            output_path=tmp_path / "out.onnx",
            model_type="unet",
            validate=False,
        )
