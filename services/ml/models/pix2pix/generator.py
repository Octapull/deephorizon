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
        self.unet = UNet(in_channels=in_channels, out_channels=out_channels)

        # Dropout applied after each upconv (decoder path).
        # Pix2Pix paper applies dropout only in decoder; we mirror that.
        self._dropout_layers = nn.ModuleList()
        if dropout > 0.0:
            for _ in range(len(self.unet.ups)):  # decoder level sayısı kadar
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
        # Encoder path — her seviyede skip connection sakla
        skips = []
        h = x
        for i, (encoder, pool) in enumerate(zip(self.unet.encoders, self.unet.pools)):
            h = encoder(h)
            skips.append(h)
            h = pool(h)

        # Bottleneck
        h = self.unet.bottleneck(h)

        # Decoder path with optional dropout
        for i, (up, decoder, skip) in enumerate(
            zip(self.unet.ups, self.unet.decoders, reversed(skips))
        ):
            h = up(h)
            if self._dropout_layers and i < len(self._dropout_layers):
                h = self._dropout_layers[i](h)
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        h = self.unet.output(h)
        h = self._tanh(h)
        return h
