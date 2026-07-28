"""Image encoding/decoding utilities for the gRPC inference server.

Handles conversion between raw bytes (PNG, JPEG, FITS) and PyTorch tensors
in ``[0, 1]`` float32 range. The gRPC contract uses ``ImagePayload``
with raw bytes + mime_type — this module is the bridge between that
contract and the model's tensor interface.

Supported formats:
    - PNG / JPEG: via ``Pillow`` (lazy import)
    - FITS: via ``astropy.io.fits`` (lazy import, optional)
    - Raw float32: ``application/octet-stream`` with explicit width/height
"""
from __future__ import annotations

import io
from typing import Literal

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Decoding: bytes → tensor
# ---------------------------------------------------------------------------


def decode_image_to_tensor(
    data: bytes,
    mime_type: str,
    width: int | None = None,
    height: int | None = None,
) -> torch.Tensor:
    """Decode raw image bytes to a float32 tensor in ``[0, 1]``.

    Args:
        data: Raw image bytes.
        mime_type: MIME type — ``"image/png"``, ``"image/jpeg"``,
            ``"image/fits"``, or ``"application/octet-stream"``.
        width: Required for ``application/octet-stream`` (raw float32).
        height: Required for ``"application/octet-stream"`` (raw float32).

    Returns:
        Tensor of shape ``(1, H, W)`` (grayscale) or ``(1, C, H, W)``
        (multi-channel), dtype ``float32``, range ``[0, 1]``.

    Raises:
        ValueError: If mime_type is unsupported or required dims are missing.
    """
    mime_type = mime_type.lower().strip()

    if mime_type in {"image/png", "image/jpeg", "image/jpg"}:
        return _decode_pil(data, mime_type)
    if mime_type == "image/fits":
        return _decode_fits(data)
    if mime_type == "application/octet-stream":
        return _decode_raw_float32(data, width=width, height=height)

    raise ValueError(
        f"Unsupported mime_type: {mime_type!r}. "
        f"Supported: image/png, image/jpeg, image/fits, application/octet-stream."
    )


def _decode_pil(data: bytes, mime_type: str) -> torch.Tensor:
    """Decode PNG/JPEG via Pillow → (1, C, H, W) float32 [0, 1]."""
    from PIL import Image  # type: ignore[import-untyped]

    image = Image.open(io.BytesIO(data))
    array = np.asarray(image, dtype=np.float32) / 255.0

    # Grayscale: (H, W) → (1, 1, H, W)
    if array.ndim == 2:
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    # RGB/RGBA: (H, W, C) → (1, C, H, W)
    else:
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    return tensor


def _decode_fits(data: bytes) -> torch.Tensor:
    """Decode FITS via astropy → (1, 1, H, W) float32 [0, 1]."""
    try:
        from astropy.io import fits  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "FITS decoding requires astropy. Install with: pip install astropy"
        ) from exc

    with fits.open(io.BytesIO(data)) as hdul:
        # Primary HDU data
        array = hdul[0].data.astype(np.float32)

    # Normalize to [0, 1] using min-max
    array_min = array.min()
    array_max = array.max()
    if array_max > array_min:
        array = (array - array_min) / (array_max - array_min)
    else:
        array = np.zeros_like(array)

    # (H, W) → (1, 1, H, W)
    if array.ndim == 2:
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    else:
        tensor = torch.from_numpy(array).unsqueeze(0)

    return tensor


def _decode_raw_float32(
    data: bytes,
    width: int | None,
    height: int | None,
) -> torch.Tensor:
    """Decode raw float32 bytes → (1, 1, H, W) float32 [0, 1]."""
    if width is None or height is None:
        raise ValueError(
            "application/octet-stream requires explicit width and height"
        )

    expected_bytes = width * height * 4  # float32 = 4 bytes
    if len(data) != expected_bytes:
        raise ValueError(
            f"Raw float32 size mismatch: got {len(data)} bytes, "
            f"expected {expected_bytes} ({width}x{height}x4)"
        )

    array = np.frombuffer(data, dtype=np.float32).reshape(height, width)
    tensor = torch.from_numpy(array.copy()).unsqueeze(0).unsqueeze(0)
    return tensor


# ---------------------------------------------------------------------------
# Encoding: tensor → bytes
# ---------------------------------------------------------------------------


def encode_tensor_to_image(
    tensor: torch.Tensor,
    output_format: Literal["png", "jpeg", "fits", "raw"] = "png",
) -> tuple[bytes, str, int, int]:
    """Encode a tensor to raw image bytes.

    Args:
        tensor: Image tensor of shape ``(1, C, H, W)`` or ``(C, H, W)``,
            dtype ``float32``, range ``[0, 1]``.
        output_format: Target format — ``"png"``, ``"jpeg"``, ``"fits"``,
            or ``"raw"`` (float32 bytes).

    Returns:
        Tuple of ``(data, mime_type, width, height)``.

    Raises:
        ValueError: If output_format is unsupported.
    """
    output_format = output_format.lower().strip()

    # Squeeze batch dim if present
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    # Clamp to [0, 1] and convert to numpy
    array = tensor.detach().cpu().clamp(0.0, 1.0).numpy()

    height, width = array.shape[-2], array.shape[-1]

    if output_format == "png":
        return _encode_pil(array, "PNG", "image/png", width, height)
    if output_format in {"jpeg", "jpg"}:
        return _encode_pil(array, "JPEG", "image/jpeg", width, height)
    if output_format == "fits":
        return _encode_fits(array, width, height)
    if output_format == "raw":
        return _encode_raw_float32(array, width, height)

    raise ValueError(
        f"Unsupported output_format: {output_format!r}. "
        f"Supported: png, jpeg, fits, raw."
    )


def _encode_pil(
    array: np.ndarray,
    pil_format: str,
    mime_type: str,
    width: int,
    height: int,
) -> tuple[bytes, str, int, int]:
    """Encode numpy array via Pillow → bytes."""
    from PIL import Image  # type: ignore[import-untyped]

    # (C, H, W) → (H, W, C) for Pillow
    if array.ndim == 3:
        array_hwc = array.transpose(1, 2, 0)
    else:
        array_hwc = array

    # Scale to [0, 255] uint8
    array_uint8 = (array_hwc * 255.0).clip(0, 255).astype(np.uint8)

    image = Image.fromarray(array_uint8)
    buffer = io.BytesIO()
    image.save(buffer, format=pil_format)
    return buffer.getvalue(), mime_type, width, height


def _encode_fits(
    array: np.ndarray,
    width: int,
    height: int,
) -> tuple[bytes, str, int, int]:
    """Encode numpy array as FITS → bytes."""
    try:
        from astropy.io import fits  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "FITS encoding requires astropy. Install with: pip install astropy"
        ) from exc

    # (C, H, W) → (H, W) for FITS primary HDU
    if array.ndim == 3:
        array_2d = array[0]
    else:
        array_2d = array

    buffer = io.BytesIO()
    fits.writeto(buffer, array_2d.astype(np.float32), overwrite=True)
    return buffer.getvalue(), "image/fits", width, height


def _encode_raw_float32(
    array: np.ndarray,
    width: int,
    height: int,
) -> tuple[bytes, str, int, int]:
    """Encode numpy array as raw float32 bytes."""
    if array.ndim == 3:
        array_2d = array[0]
    else:
        array_2d = array

    return array_2d.astype(np.float32).tobytes(), "application/octet-stream", width, height
