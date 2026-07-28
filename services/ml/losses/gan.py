"""Adversarial (GAN) losses for conditional image generation.

Provides two loss modes:

- ``"vanilla"``: Standard binary cross-entropy GAN (Goodfellow et al., 2014).
- ``"lsgan"``: Least-squares GAN (Mao et al., 2017) — more stable training,
  higher quality samples, no vanishing gradients.

Both losses operate on the discriminator's patch-level output and are
designed for use with the ``PatchDiscriminator`` from
``services.ml.models.pix2pix``.

Reference:
    Goodfellow, I. et al. (2014). "Generative Adversarial Nets." NIPS.
    Mao, X. et al. (2017). "Least Squares Generative Adversarial Networks."
    ICCV.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorAdversarialLoss(nn.Module):
    """Adversarial loss for the generator.

    Encourages the generator to produce images that the discriminator
    classifies as real. For ``"vanilla"`` mode, minimizes
    ``-log D(G(z))`` (log(D) saturates near 0, so we use the
    log-prob trick). For ``"lsgan"`` mode, minimizes
    ``(D(G(z)) - 1)^2``.

    Args:
        mode: ``"vanilla"`` (BCE) or ``"lsgan"`` (MSE).
        target_real_label: Label value for "real" (default ``1.0``).
    """

    def __init__(
        self,
        mode: Literal["vanilla", "lsgan"] = "vanilla",
        target_real_label: float = 1.0,
    ) -> None:
        super().__init__()

        if mode not in {"vanilla", "lsgan"}:
            raise ValueError(f"mode must be 'vanilla' or 'lsgan', got {mode!r}")

        self.mode = mode
        self.target_real_label = target_real_label

    def forward(self, disc_output: torch.Tensor) -> torch.Tensor:
        """Compute generator adversarial loss.

        Args:
            disc_output: Discriminator logits/probabilities for
                generated (fake) images, shape ``(B, 1, H, W)``.

        Returns:
            Scalar loss tensor.
        """
        if self.mode == "vanilla":
            # Vanilla GAN: -log(D(G(z))) — equivalent to BCE with real labels
            target = torch.full_like(disc_output, self.target_real_label)
            return F.binary_cross_entropy(disc_output, target)

        # LSGAN: (D(G(z)) - 1)^2
        target = torch.full_like(disc_output, self.target_real_label)
        return F.mse_loss(disc_output, target)


class DiscriminatorAdversarialLoss(nn.Module):
    """Adversarial loss for the discriminator.

    Trains the discriminator to distinguish real from fake images.
    For ``"vanilla"`` mode, uses BCE with real/fake labels. For
    ``"lsgan"`` mode, uses MSE with target ``1.0`` for real and
    ``0.0`` for fake.

    Args:
        mode: ``"vanilla"`` (BCE) or ``"lsgan"`` (MSE).
        target_real_label: Label for real images (default ``1.0``).
        target_fake_label: Label for fake images (default ``0.0``).
    """

    def __init__(
        self,
        mode: Literal["vanilla", "lsgan"] = "vanilla",
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
    ) -> None:
        super().__init__()

        if mode not in {"vanilla", "lsgan"}:
            raise ValueError(f"mode must be 'vanilla' or 'lsgan', got {mode!r}")

        self.mode = mode
        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label

    def forward(
        self,
        disc_real: torch.Tensor,
        disc_fake: torch.Tensor,
    ) -> torch.Tensor:
        """Compute discriminator adversarial loss.

        Args:
            disc_real: Discriminator output for real images,
                shape ``(B, 1, H, W)``.
            disc_fake: Discriminator output for fake (generated) images,
                same shape as ``disc_real``.

        Returns:
            Scalar loss tensor (sum of real + fake halves).
        """
        if self.mode == "vanilla":
            real_target = torch.full_like(disc_real, self.target_real_label)
            fake_target = torch.full_like(disc_fake, self.target_fake_label)
            real_loss = F.binary_cross_entropy(disc_real, real_target)
            fake_loss = F.binary_cross_entropy(disc_fake, fake_target)
            return (real_loss + fake_loss) * 0.5

        # LSGAN: (D(x) - 1)^2 + D(G(z))^2
        real_target = torch.full_like(disc_real, self.target_real_label)
        fake_target = torch.full_like(disc_fake, self.target_fake_label)
        real_loss = F.mse_loss(disc_real, real_target)
        fake_loss = F.mse_loss(disc_fake, fake_target)
        return (real_loss + fake_loss) * 0.5
