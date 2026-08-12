"""Relativistic Average GAN (RaGAN) Discriminator — ESRGAN için.

RaGAN, standart GAN'dan farklı olarak gerçek/fake ayrımını **"gerçek
ortalamadan ne kadar uzak"** olarak öğrenir. Bu, eğitim kararlılığını
artırır ve daha yüksek kaliteli üretim sağlar.

Matematiksel formülasyon:
    D_ra(x_r, x_f) = sigmoid(C(x_r) - E[C(x_f)])
    D_ra(x_f, x_r) = sigmoid(C(x_f) - E[C(x_r)])

Burada C(x) discriminator'ın patch-level çıktısıdır. Generator loss:
    L_G = -E[log(D_ra(x_f, x_r))]
    L_D = -E[log(1 - D_ra(x_f, x_r))]

Bu implementasyon PatchGAN mimarisini kullanır (ESRGAN makalesi):
- 5 katmanlı Conv(stride=2) → LeakyReLU(0.2)
- Son katmanda sigmoid YOK (LSGAN + RaGAN birlikte)

Referans:
    Jolicoeur-Martineau, A. (2018). "The Relativistic Discriminator:
    A Key Element Missing from Standard GAN." arXiv:1807.00734.
    Wang, X. et al. (2018). "ESRGAN: Enhanced Super-Resolution
    Generative Adversarial Networks." ECCV Workshops.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Conv-BN-LeakyReLU bloğu (RaGAN discriminator için)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size=4, stride=stride, padding=1
            )
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class RaDiscriminator(nn.Module):
    """Relativistic Average PatchGAN discriminator — ESRGAN için.

    Standart PatchGAN'den farkları:
    - ``n_layers=5`` (ESRGAN makalesi, Pix2Pix'in 3'ünden fazla)
    - Sigmoid YOK (LSGAN + RaGAN birlikte kullanılır)
    - ``forward(real, fake)`` → ``(D_real_rel, D_fake_rel)`` tuple döner

    Args:
        in_channels: Birleştirilmiş giriş kanal sayısı (condition + target).
        base_channels: İlk conv katmanının kanal sayısı (default 64).
        max_channels: Üst kanal sınırı (default 512).
        n_layers: Conv katmanı sayısı (default 5, ESRGAN).
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        max_channels: int = 512,
        n_layers: int = 5,
    ) -> None:
        super().__init__()

        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.max_channels = max_channels
        self.n_layers = n_layers

        layers: list[nn.Module] = []

        # İlk katman: BN yok (Pix2Patch standart)
        layers.append(_ConvBlock(in_channels, base_channels, stride=2, use_bn=False))

        # Orta katmanlar: stride=2, kanalları iki katına çıkar
        in_ch = base_channels
        for _ in range(1, n_layers - 1):
            out_ch = min(in_ch * 2, max_channels)
            layers.append(_ConvBlock(in_ch, out_ch, stride=2, use_bn=True))
            in_ch = out_ch

        # Son-öncesi katman: stride=1
        out_ch = min(in_ch * 2, max_channels)
        layers.append(_ConvBlock(in_ch, out_ch, stride=1, use_bn=True))
        in_ch = out_ch

        # Çıkış katmanı: 1 kanal, sigmoid yok
        layers.append(nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(
        self,
        condition: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Standart PatchGAN forward — (condition, target) birleştirilir.

        Args:
            condition: Koşul görüntüsü, shape ``(B, C_cond, H, W)``.
            target: Hedef görüntü, shape ``(B, C_tgt, H, W)``.

        Returns:
            Patch-level logit haritası, shape ``(B, 1, H_out, W_out)``.
        """
        if condition.shape[2:] != target.shape[2:]:
            raise ValueError(
                f"Spatial size mismatch: condition {condition.shape[2:]}"
                f" vs target {target.shape[2:]}"
            )

        x = torch.cat([condition, target], dim=1)
        return self.model(x)

    def relativistic_logits(
        self,
        real_output: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Relativistic average logits hesapla.

        D_ra(x_r, x_f) = C(x_r) - mean(C(x_f))
        D_ra(x_f, x_r) = C(x_f) - mean(C(x_r))

        Args:
            real_output: Gerçek görüntüler için discriminator çıktısı.
            fake_output: Üretilmiş görüntüler için discriminator çıktısı.

        Returns:
            Tuple of ``(D_real_rel, D_fake_rel)``.
        """
        # Fake ortalaması (gerçek batch üzerinden)
        fake_mean = fake_output.mean(dim=0, keepdim=True)
        real_mean = real_output.mean(dim=0, keepdim=True)

        # Relativistic logit'ler
        D_real_rel = real_output - fake_mean
        D_fake_rel = fake_output - real_mean

        return D_real_rel, D_fake_rel
