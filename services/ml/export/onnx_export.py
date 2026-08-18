"""ONNX export utilities for DeepHorizon models.

Exports trained PyTorch models (U-Net, Pix2PixGenerator) to ONNX format
for deployment in the gRPC inference server. Supports:

- Loading checkpoints (U-Net or Pix2Pix GAN format)
- Tracing the model with a dummy input
- ONNX export with configurable opset and dynamic axes
- Validation: re-load the ONNX model and compare outputs
- Metadata sidecar (JSON) with model info, input/output specs

The exported ONNX model is consumed by the Go gRPC server via
``onnxruntime-go`` for low-latency inference.

Reference:
    ONNX: https://onnx.ai/
    torch.onnx: https://pytorch.org/docs/stable/onnx.html
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from services.ml.models.pix2pix import Pix2PixGenerator
from services.ml.models.unet import UNet

# ---------------------------------------------------------------------------
# Metadata dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportMetadata:
    """Sidecar metadata for an exported ONNX model.

    Stored as ``<model_name>.json`` next to the ``.onnx`` file.
    """

    model_name: str
    model_type: Literal["unet", "pix2pix_generator"]
    onnx_opset: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_dtype: str
    output_dtype: str
    checkpoint_path: str
    checkpoint_epoch: int
    val_loss: float
    psnr: float
    ssim: float
    export_timestamp: str


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _load_unet_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[UNet, dict]:
    """Load a U-Net model from a standard checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _load_pix2pix_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[Pix2PixGenerator, dict]:
    """Load a Pix2PixGenerator from a GAN checkpoint.

    GAN checkpoints (from ``gan_train.py``) store the generator under
    ``generator_state_dict``. Falls back to ``model_state_dict`` for
    standard checkpoints.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Determine in/out channels from checkpoint shape
    state_dict_key = (
        "generator_state_dict"
        if "generator_state_dict" in checkpoint
        else "model_state_dict"
    )
    state_dict = checkpoint[state_dict_key]

    # Infer in_channels from first conv weight shape: (out, in, k, k)
    first_weight = state_dict["unet.enc1.block.0.weight"]
    in_channels = first_weight.shape[1]

    # Infer out_channels from output conv weight shape: (out, in, 1, 1)
    output_weight = state_dict["unet.output.weight"]
    out_channels = output_weight.shape[0]

    model = Pix2PixGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        dropout=0.0,  # eval mode — no dropout
        use_tanh=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def load_model_from_checkpoint(
    checkpoint_path: Path | str,
    model_type: Literal["unet", "pix2pix_generator"],
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict]:
    """Load a model from a checkpoint file.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint.
        model_type: ``"unet"`` or ``"pix2pix_generator"``.
        device: Device to map tensors to. Defaults to CPU.

    Returns:
        Tuple of ``(model, checkpoint_dict)``. The model is in eval mode.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cpu")

    if model_type == "unet":
        return _load_unet_from_checkpoint(checkpoint_path, device)
    if model_type == "pix2pix_generator":
        return _load_pix2pix_from_checkpoint(checkpoint_path, device)

    raise ValueError(
        f"Unknown model_type: {model_type!r}. "
        f"Supported: 'unet', 'pix2pix_generator'."
    )


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path | str,
    input_shape: tuple[int, ...] = (1, 1, 256, 256),
    onnx_opset: int = 17,
    dynamic_axes: bool = True,
    validate: bool = True,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> Path:
    """Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model in eval mode.
        output_path: Where to write the ``.onnx`` file.
        input_shape: Dummy input shape ``(B, C, H, W)``.
        onnx_opset: ONNX opset version (default 17, supported by
            onnxruntime-go).
        dynamic_axes: If ``True``, allow variable batch size and spatial
            dimensions in the exported model.
        validate: If ``True``, re-load the ONNX model and compare outputs
            against the PyTorch model.
        atol: Absolute tolerance for validation.
        rtol: Relative tolerance for validation.

    Returns:
        Path to the exported ``.onnx`` file.

    Raises:
        RuntimeError: If validation fails (PyTorch vs ONNX outputs differ).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(*input_shape)

    # Dynamic axes: batch + spatial dims variable, channel fixed
    dynamic_axes_dict: dict[str, dict[int, str]] | None = None
    if dynamic_axes:
        dynamic_axes_dict = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes_dict,
        opset_version=onnx_opset,
        do_constant_folding=True,
    )

    # Validate
    if validate:
        _validate_onnx_export(
            model=model,
            onnx_path=output_path,
            dummy_input=dummy_input,
            atol=atol,
            rtol=rtol,
        )

    return output_path


def _validate_onnx_export(
    model: torch.nn.Module,
    onnx_path: Path,
    dummy_input: torch.Tensor,
    atol: float,
    rtol: float,
) -> None:
    """Validate ONNX export by comparing PyTorch vs ONNX outputs.

    Uses ``onnxruntime`` if available; otherwise skips validation with a
    warning. Raises ``RuntimeError`` if outputs differ beyond tolerance.
    """
    try:
        import numpy as np
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError:
        # onnxruntime not installed — skip validation
        return

    # PyTorch reference
    model.eval()
    with torch.no_grad():
        torch_output = model(dummy_input).cpu().numpy()

    # ONNX inference
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"input": dummy_input.numpy()})[0]

    if not np.allclose(torch_output, onnx_output, atol=atol, rtol=rtol):
        max_diff = float(np.abs(torch_output - onnx_output).max())
        raise RuntimeError(
            f"ONNX validation failed: max diff = {max_diff:.6f} "
            f"(atol={atol}, rtol={rtol})"
        )


# ---------------------------------------------------------------------------
# Metadata sidecar
# ---------------------------------------------------------------------------


def save_metadata(
    metadata: ExportMetadata,
    output_path: Path | str,
) -> Path:
    """Save export metadata as a JSON sidecar file.

    Args:
        metadata: ExportMetadata dataclass.
        output_path: Path to the ``.onnx`` file. The metadata is written
            to ``<output_path>.json``.

    Returns:
        Path to the ``.json`` metadata file.
    """
    output_path = Path(output_path)
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)

    return metadata_path


def load_metadata(metadata_path: Path | str) -> ExportMetadata:
    """Load export metadata from a JSON sidecar file.

    JSON tuple'ları list olarak saklar; burada ``input_shape`` ve
    ``output_shape`` alanlarını tekrar tuple'a çeviriyoruz.
    """
    metadata_path = Path(metadata_path)
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # JSON tuple'ları list olarak saklar; dataclass tuple bekliyor
    for key in ("input_shape", "output_shape"):
        if key in data and isinstance(data[key], list):
            data[key] = tuple(data[key])

    return ExportMetadata(**data)


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------


def export_checkpoint_to_onnx(
    checkpoint_path: Path | str,
    output_path: Path | str,
    model_type: Literal["unet", "pix2pix_generator"] = "pix2pix_generator",
    input_shape: tuple[int, ...] = (1, 1, 256, 256),
    onnx_opset: int = 17,
    dynamic_axes: bool = True,
    validate: bool = True,
) -> tuple[Path, Path]:
    """End-to-end export: checkpoint → ONNX + metadata.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint.
        output_path: Where to write the ``.onnx`` file.
        model_type: ``"unet"`` or ``"pix2pix_generator"``.
        input_shape: Dummy input shape ``(B, C, H, W)``.
        onnx_opset: ONNX opset version.
        dynamic_axes: Allow variable batch/spatial dims.
        validate: Validate PyTorch vs ONNX outputs.

    Returns:
        Tuple of ``(onnx_path, metadata_path)``.
    """
    from datetime import datetime, timezone

    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    # Load model
    device = torch.device("cpu")
    model, checkpoint = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=device,
    )

    # Export
    onnx_path = export_to_onnx(
        model=model,
        output_path=output_path,
        input_shape=input_shape,
        onnx_opset=onnx_opset,
        dynamic_axes=dynamic_axes,
        validate=validate,
    )

    # Get output shape by running a dummy forward
    with torch.no_grad():
        dummy = torch.randn(*input_shape)
        output = model(dummy)
    output_shape = tuple(output.shape)

    # Build metadata
    metadata = ExportMetadata(
        model_name=output_path.stem,
        model_type=model_type,
        onnx_opset=onnx_opset,
        input_shape=input_shape,
        output_shape=output_shape,
        input_dtype="float32",
        output_dtype="float32",
        checkpoint_path=str(checkpoint_path),
        checkpoint_epoch=int(checkpoint.get("epoch", 0)),
        val_loss=float(checkpoint.get("val_loss", 0.0)),
        psnr=float(checkpoint.get("psnr", 0.0)),
        ssim=float(checkpoint.get("ssim", 0.0)),
        export_timestamp=datetime.now(timezone.utc).isoformat(),
    )

    metadata_path = save_metadata(metadata, onnx_path)

    return onnx_path, metadata_path
