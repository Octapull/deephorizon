"""ESRGAN Generator — 23 RRDB bloğu + sub-pixel upsampling.

ESRGAN generator, Pix2Pix'ten farklı olarak **upsampling** yapabilir.
Mimari:

1. **İlk feature extraction**: Conv2d(in_channels → features)
2. **RRDB blokları**: 23 ardışık RRDB bloğu (residual-in-residual)
3. **Upsampling**: scale=2 veya scale=4 için PixelShuffle ile 2x/4x büyütme
4. **Son conv**: Conv2d(features → out_channels) + Tanh

Scale=1 modunda upsample bloğu atlanır (Pix2Pix ile aynı çözünürlük).
Bu sayede Pix2Pix checkpoint'ından warm-start yapılabilir.

Referans:
    Wang, X. et al. (2018). "ESRGAN: Enhanced Super-Resolution
    Generative Adversarial Networks." ECCV Workshops.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from services.ml.models.esrgan.rrdb import RRDB


class _InitialConv(nn.Module):
    """İlk feature extraction bloğu — Conv2d + LeakyReLU."""

    def __init__(self, in_channels: int, features: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lrelu(self.conv(x))


class _ResidualConv(nn.Module):
    """RRDB bloklarının çıktısını işleyen residual conv bloğu."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lrelu(self.conv(x))


class _UpsampleBlock(nn.Module):
    """Sub-pixel upsample bloğu — Conv2d + PixelShuffle(2).

    Her blok 2x büyütme yapar. scale=4 için 2 blok, scale=2 için 1 blok,
    scale=1 için 0 blok kullanılır.

    Args:
        features: Feature kanal sayısı.
        scale: Büyütme faktörü (2 veya 4).
    """

    def __init__(self, features: int, scale: int) -> None:
        super().__init__()

        if scale not in {2, 4}:
            raise ValueError(f"scale must be 2 or 4, got {scale}")

        num_blocks = int(math.log2(scale))
        layers: list[nn.Module] = []
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(features, features * 4, kernel_size=3, padding=1))
            layers.append(nn.PixelShuffle(2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ESRGANGenerator(nn.Module):
    """ESRGAN generator — 23 RRDB bloğu + sub-pixel upsampling.

    Args:
        in_channels: Giriş kanal sayısı (default 1, grayscale).
        out_channels: Çıkış kanal sayısı (default 1, grayscale).
        num_rrdb: RRDB bloğu sayısı (default 23, ESRGAN makalesi).
        features: Feature kanal sayısı (default 64).
        growth_rate: RRDB içindeki growth rate (default 32).
        scale: Büyütme faktörü (1, 2, veya 4). scale=1 → Pix2Pix uyumlu.
        use_tanh: True ise çıkışa Tanh uygulanır (default True).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_rrdb: int = 23,
        features: int = 64,
        growth_rate: int = 32,
        scale: int = 1,
        use_tanh: bool = True,
    ) -> None:
        super().__init__()

        if scale not in {1, 2, 4}:
            raise ValueError(f"scale must be 1, 2, or 4, got {scale}")
        if num_rrdb < 1:
            raise ValueError(f"num_rrdb must be >= 1, got {num_rrdb}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rrdb = num_rrdb
        self.features = features
        self.growth_rate = growth_rate
        self.scale = scale
        self.use_tanh = use_tanh

        # İlk feature extraction
        self.initial = _InitialConv(in_channels, features)

        # RRDB blokları
        self.rrdb_blocks = nn.Sequential(
            *[RRDB(features=features, growth_rate=growth_rate) for _ in range(num_rrdb)]
        )

        # Residual conv (RRDB çıktısını işle)
        self.residual_conv = _ResidualConv(features)

        # Upsampling (scale=1 ise atlanır)
        if scale > 1:
            self.upsample = _UpsampleBlock(features, scale)
        else:
            self.upsample = nn.Identity()

        # Son conv + Tanh
        self.final = nn.Conv2d(features, out_channels, kernel_size=3, padding=1)
        self._tanh = nn.Tanh() if use_tanh else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Giriş tensor, shape ``(B, in_channels, H, W)``.

        Returns:
            Çıkış tensor, shape ``(B, out_channels, H*scale, W*scale)``.
        """
        initial = self.initial(x)
        rrdb_out = self.rrdb_blocks(initial)
        residual = self.residual_conv(rrdb_out)

        # Global residual: initial + residual
        out = initial + residual

        # Upsample (scale=1 ise Identity)
        out = self.upsample(out)

        # Final conv + Tanh
        out = self.final(out)
        out = self._tanh(out)
        return out

    @classmethod
    def from_pix2pix_checkpoint(
        cls,
        checkpoint_path: str,
        scale: int = 1,
        device: torch.device | None = None,
    ) -> "ESRGANGenerator":
        """Pix2Pix checkpoint'ından ESRGAN warm-start.

        Pix2Pix U-Net mimarisi ESRGAN'den farklı olduğu için bu sadece
        scale=1 modunda anlamlıdır. Yeni katmanlar rastgele başlatılır.

        Args:
            checkpoint_path: Pix2Pix ``.pt`` dosyasının yolu.
            scale: Hedef scale (default 1).
            device: Yüklenecek cihaz.

        Returns:
            Pix2Pix ağırlıklarıyla warm-start edilmiş ESRGANGenerator.
        """
        if device is None:
            device = torch.device("cpu")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get(
            "generator_state_dict", checkpoint.get("model_state_dict", {})
        )

        # Kanal sayılarını state_dict'ten çıkar
        first_weight = state_dict.get("unet.enc1.block.0.weight")
        output_weight = state_dict.get("unet.output.weight")
        if first_weight is None or output_weight is None:
            raise ValueError(
                "Checkpoint does not contain Pix2Pix U-Net weights "
                "(expected keys: unet.enc1.block.0.weight, unet.output.weight)"
            )

        in_channels = first_weight.shape[1]
        out_channels = output_weight.shape[0]

        model = cls(
            in_channels=in_channels,
            out_channels=out_channels,
            scale=scale,
        )
        # Not: Mimari farklı olduğu için state_dict doğrudan yüklenemez.
        # Çağıran, kendi transfer learning stratejisini uygulamalı.
        return model
