"""Restormer — Efficient Transformer for High-Resolution Image Restoration.

This package implements the Restormer architecture (Zamir et al., 2022),
a transformer-based encoder-decoder designed for high-resolution image
restoration tasks (denoising, deraining, deblurring, super-resolution).

Key components:

- ``mdta``: Multi-Dconv Head Transposed Self-Attention — channel-wise
  attention with depth-wise conv preprocessing. Reduces attention cost
  from O(N²) to O(C²) compared to standard ViT.
- ``gdfn``: Gated-Dconv Feed-forward Network — gated FFN with depth-wise
  conv for local context. Controls information flow via element-wise
  gating.
- ``restormer``: Restormer — 4-level encoder-decoder transformer that
  combines MDTA and GDFN blocks with skip connections.

Reference:
    Zamir, S. W., Arora, A., Khan, S., Hayat, M., Khan, F. S., Yang, M.-H.,
    Shao, L. (2022). "Restormer: Efficient Transformer for High-Resolution
    Image Restoration." CVPR.
"""

from __future__ import annotations

from services.ml.models.restormer.gdfn import GDFN
from services.ml.models.restormer.mdta import MDTA
from services.ml.models.restormer.restormer import (
    Downsample,
    Restormer,
    TransformerBlock,
    Upsample,
)

__all__ = [
    "Downsample",
    "GDFN",
    "MDTA",
    "Restormer",
    "TransformerBlock",
    "Upsample",
]


def build_restormer(
    in_channels: int = 1,
    out_channels: int = 1,
    dim: int = 48,
    num_blocks: list[int] | None = None,
    num_heads: list[int] | None = None,
    expansion_factor: float = 2.66,
) -> Restormer:
    """Factory function to build a Restormer model with default settings.

    This is the recommended way to instantiate Restormer in training
    scripts and inference pipelines. It mirrors the ``build_pix2pix``
    and ``build_esrgan`` factory functions used elsewhere in the project.

    Args:
        in_channels: Number of input channels (default 1, grayscale).
        out_channels: Number of output channels (default 1, grayscale).
        dim: Base channel count for level 1 (default 48). Subsequent
            levels double: 48, 96, 192, 384.
        num_blocks: Transformer blocks per level (default [4, 6, 6, 8]).
        num_heads: Attention heads per level (default [1, 2, 4, 8]).
        expansion_factor: GDFN expansion ratio (default 2.66).

    Returns:
        A configured ``Restormer`` model instance.
    """
    return Restormer(
        in_channels=in_channels,
        out_channels=out_channels,
        dim=dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        expansion_factor=expansion_factor,
    )
