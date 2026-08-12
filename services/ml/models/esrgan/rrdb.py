"""Residual-in-Residual Dense Block (RRDB) — ESRGAN'in temel yapı taşı.

RRDB, ESRGAN makalesinin (Wang et al., 2018) temel yapı bloğudur. Üç
iç içe residual yapıdan oluşur:

1. **Dense layer**: 4 katmanlı (Conv-LeakyReLU) yoğun bağlantı bloğu.
   Her katman, önceki tüm katmanların feature map'lerini toplar
   (growth_rate kadar yeni kanal ekler).
2. **Residual block**: 3 dense layer'ın toplamı + residual scaling (β=0.2)
   + skip connection. Residual scaling, eğitim kararlılığını artırır.
3. **Residual-in-Residual**: Residual block'un çıktısı, girişe eklenir
   (β=0.2 ile ölçeklenmiş).

Bu hiyerarşik residual yapı, ESRGAN'in derin ağlarda (23 RRDB bloğu)
kararlı bir şekilde eğitilmesini sağlar.

Referans:
    Wang, X. et al. (2018). "ESRGAN: Enhanced Super-Resolution
    Generative Adversarial Networks." ECCV Workshops.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    """4 katmanlı yoğun bağlantı bloğu (Conv-LeakyReLU).

    Her Conv2d katmanı, önceki tüm katmanların çıktılarını birleştirir
    (concatenate yerine sum kullanır — daha kararlı eğitim). Growth
    rate kadar yeni kanal eklenir.

    Args:
        in_channels: Giriş kanal sayısı.
        growth_rate: Her katmanın eklediği kanal sayısı (default 32).
    """

    def __init__(self, in_channels: int, growth_rate: int = 32) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels + growth_rate, growth_rate, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels + 2 * growth_rate, growth_rate, kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels + 3 * growth_rate, growth_rate, kernel_size=3, padding=1
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — 4 katmanlı yoğun bağlantı.

        Args:
            x: Giriş tensor, shape ``(B, in_channels, H, W)``.

        Returns:
            Çıkış tensor, shape ``(B, growth_rate, H, W)``.
        """
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        # Son katmanın çıktısı — growth_rate kanal
        return x4


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block — ESRGAN'in temel yapı taşı.

    3 DenseLayer + residual scaling (β=0.2) + skip connection. Residual
    scaling, derin ağlarda eğitim kararlılığını artırır (ESRGAN makalesi).

    Args:
        features: Feature kanal sayısı (default 32).
        growth_rate: DenseLayer'ın her katmanında eklenen kanal sayısı
            (default 32).
        num_dense_layers: DenseLayer sayısı (default 3).
        residual_scale: Residual scaling faktörü β (default 0.2).
    """

    def __init__(
        self,
        features: int = 32,
        growth_rate: int = 32,
        num_dense_layers: int = 3,
        residual_scale: float = 0.2,
    ) -> None:
        super().__init__()

        if num_dense_layers < 1:
            raise ValueError(f"num_dense_layers must be >= 1, got {num_dense_layers}")
        if not 0.0 <= residual_scale <= 1.0:
            raise ValueError(f"residual_scale must be in [0, 1], got {residual_scale}")

        self.features = features
        self.growth_rate = growth_rate
        self.num_dense_layers = num_dense_layers
        self.residual_scale = residual_scale

        # DenseLayer'lar — her biri features → growth_rate kanal üretir
        self.dense_layers = nn.ModuleList()
        for _ in range(num_dense_layers):
            self.dense_layers.append(DenseLayer(features, growth_rate))

        # Son fusion conv: growth_rate → features
        self.fusion = nn.Conv2d(growth_rate, features, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — residual-in-residual.

        Args:
            x: Giriş tensor, shape ``(B, features, H, W)``.

        Returns:
            Çıkış tensor, shape ``(B, features, H, W)``.
        """
        residual = x

        # Dense layer'ları sırayla uygula
        out = x
        for dense_layer in self.dense_layers:
            out = dense_layer(out)

        # Fusion conv: growth_rate → features
        out = self.fusion(out)

        # Residual scaling + skip connection
        return residual + self.residual_scale * out
