"""Pix2Pix generator — U-Net wrapper for conditional image generation.

The generator is an encoder-decoder with skip connections (U-Net) that
maps a degraded input image to a clean output image. Dropout is applied
in the decoder (between upconv and concat) to introduce stochasticity,
following Isola et al. (2017).

Reference:
    Isola, P., Zhu, J.-Y., Zhou, T., Efros, A. A. (2017).
    "Image-to-Image Translation with Conditional Adversarial Networks."
    CVPR.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from services.ml.models.unet import UNet


class Pix2PixGenerator(nn.Module):
    """U-Net based generator for Pix2Pix.

    Wraps the deterministic ``UNet`` and adds optional dropout layers in
    the decoder path. The output is passed through ``Tanh`` to constrain
    pixel values to ``[-1, 1]`` — the standard Pix2Pix output range.

    Args:
        in_channels: Number of input channels (degraded image).
        out_channels: Number of output channels (generated image).
        dropout: Dropout probability in the decoder. ``0.0`` disables it.
        use_tanh: If ``True``, apply ``Tanh`` to the output. Pix2Pix
            training expects ``[-1, 1]``; inference pipelines that work
            in ``[0, 1]`` may set this to ``False``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        dropout: float = 0.5,
        use_tanh: bool = True,
    ) -> None:
        super().__init__()

        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.use_tanh = use_tanh

        # Core U-Net (encoder-decoder with skip connections)
        self.unet = UNet()

        # Override first conv to accept arbitrary in_channels.
        # The default UNet is hard-coded to 1 input channel; we replace
        # enc1's first conv to support multi-channel inputs while keeping
        # the rest of the architecture intact.
        if in_channels != 1:
            self.unet.enc1 = self._make_input_block(in_channels, 64)

        # Override output conv for arbitrary out_channels.
        if out_channels != 1:
            self.unet.output = nn.Conv2d(64, out_channels, kernel_size=1)

        # Dropout applied after each upconv (decoder path).
        # Pix2Pix paper applies dropout only in decoder; we mirror that.
        self._dropout_layers = nn.ModuleList()
        if dropout > 0.0:
            for _ in range(4):  # 4 decoder levels (up1..up4)
                self._dropout_layers.append(nn.Dropout2d(p=dropout))

        # Output activation — Tanh constrains to [-1, 1].
        self._tanh = nn.Tanh() if use_tanh else nn.Identity()

    @staticmethod
    def _make_input_block(in_channels: int, out_channels: int) -> nn.Module:
        """Build a DoubleConv block with custom input channels."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net with decoder dropout.

        Args:
            x: Input tensor of shape ``(B, in_channels, H, W)``.

        Returns:
            Generated image of shape ``(B, out_channels, H, W)``,
            optionally passed through ``Tanh``.
        """
        # Encoder path
        x1 = self.unet.enc1(x)
        x2 = self.unet.pool1(x1)
        x2 = self.unet.enc2(x2)
        x3 = self.unet.pool2(x2)
        x3 = self.unet.enc3(x3)
        x4 = self.unet.pool3(x3)
        x4 = self.unet.enc4(x4)
        x5 = self.unet.pool4(x4)
        x5 = self.unet.bottleneck(x5)

        # Decoder path with optional dropout
        x = self.unet.up1(x5)
        if self._dropout_layers:
            x = self._dropout_layers[0](x)
        x = torch.cat([x, x4], dim=1)
        x = self.unet.dec1(x)

        x = self.unet.up2(x)
        if self._dropout_layers:
            x = self._dropout_layers[1](x)
        x = torch.cat([x, x3], dim=1)
        x = self.unet.dec2(x)

        x = self.unet.up3(x)
        if self._dropout_layers:
            x = self._dropout_layers[2](x)
        x = torch.cat([x, x2], dim=1)
        x = self.unet.dec3(x)

        x = self.unet.up4(x)
        if self._dropout_layers:
            x = self._dropout_layers[3](x)
        x = torch.cat([x, x1], dim=1)
        x = self.unet.dec4(x)

        x = self.unet.output(x)
        x = self._tanh(x)
        return x
