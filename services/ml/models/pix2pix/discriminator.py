"""PatchGAN discriminator — 70x70 receptive field Markovian classifier.

The discriminator classifies each NxN patch as real or fake rather than
the whole image. This is equivalent to a "texture loss" and is the
standard Pix2Pix discriminator (Isola et al., 2017).

Architecture (C64 → C128 → C256 → C512):
    Conv(stride=2) → LeakyReLU(0.2)
    Conv(stride=2) → BN → LeakyReLU(0.2)
    Conv(stride=2) → BN → LeakyReLU(0.2)
    Conv(stride=1) → BN → LeakyReLU(0.2)
    Conv(stride=1) → output (1-channel prediction map)

The receptive field of this stack is 70x70 for 256x256 inputs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator for Pix2Pix.

    Takes a concatenated ``(condition, target)`` pair and outputs a
    spatial map of patch-level real/fake predictions.

    Args:
        in_channels: Channels of the concatenated input
            (condition_channels + target_channels). For grayscale
            image-to-image, this is typically ``2`` (1 + 1).
        base_channels: Channel width of the first conv layer (C64).
            Subsequent layers double this width up to ``max_channels``.
        max_channels: Upper bound on channel width (C512 in the paper).
        n_layers: Number of conv layers in the discriminator body.
            ``3`` gives the standard C64-C128-C256-C512 stack.
        use_sigmoid: If ``True``, apply ``Sigmoid`` to the output.
            Pix2Pix with LSGAN/BCE-only loss may set this to ``False``
            and rely on the loss function for stability.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        max_channels: int = 512,
        n_layers: int = 3,
        use_sigmoid: bool = True,
    ) -> None:
        super().__init__()

        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.max_channels = max_channels
        self.n_layers = n_layers
        self.use_sigmoid = use_sigmoid

        layers: list[nn.Module] = []

        # First layer: no BatchNorm (per Isola et al.)
        layers.append(
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Middle layers: stride=2, doubling channels up to max_channels
        in_ch = base_channels
        for n in range(1, n_layers):
            out_ch = min(in_ch * 2, max_channels)
            layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch

        # Penultimate layer: stride=1, channels = max_channels
        out_ch = min(in_ch * 2, max_channels)
        layers.append(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        in_ch = out_ch

        # Output layer: 1-channel prediction map
        layers.append(
            nn.Conv2d(
                in_ch,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            )
        )

        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(
        self,
        condition: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Classify each patch of the (condition, target) pair.

        Args:
            condition: Conditional input, shape ``(B, C_cond, H, W)``.
            target: Target image, shape ``(B, C_tgt, H, W)``.

        Returns:
            Prediction map of shape ``(B, 1, H_out, W_out)`` where each
            spatial entry is the real/fake logit/probability for the
            corresponding receptive field patch.
        """
        if condition.shape[2:] != target.shape[2:]:
            raise ValueError(
                f"Spatial size mismatch: condition {condition.shape[2:]}"
                f" vs target {target.shape[2:]}"
            )

        x = torch.cat([condition, target], dim=1)
        return self.model(x)
