"""Combined loss for Pix2Pix training.

Aggregates pixel, perceptual, adversarial, and physics losses with
configurable weights. This is the standard Pix2Pix objective:

    L_total = λ_pixel * L_pixel
            + λ_perceptual * L_perceptual
            + λ_adversarial * L_adv
            + λ_physics * L_physics

The pixel loss is always present (default L1). The other components
are optional and activated by setting their weight > 0.

Reference:
    Isola, P. et al. (2017). "Image-to-Image Translation with
    Conditional Adversarial Networks." CVPR.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from services.ml.losses.gan import GeneratorAdversarialLoss
from services.ml.losses.perceptual import PerceptualLoss
from services.ml.losses.physics import PhysicsLoss


class CombinedLoss(nn.Module):
    """Weighted sum of pixel + perceptual + adversarial + physics losses.

    Args:
        pixel_weight: Weight for the pixel reconstruction loss (L1 by default).
        perceptual_weight: Weight for the VGG perceptual loss. ``0.0`` disables.
        adversarial_weight: Weight for the generator adversarial loss.
            ``0.0`` disables.
        physics_weight: Weight for the physics-informed loss. ``0.0`` disables.
        pixel_loss: Name of the pixel loss — ``"l1"``, ``"mse"``, or
            ``"smooth_l1"``.
        gan_mode: ``"vanilla"`` (BCE) or ``"lsgan"`` (MSE).
        perceptual_layer: VGG19 layer for perceptual loss.
    """

    def __init__(
        self,
        pixel_weight: float = 100.0,
        perceptual_weight: float = 0.0,
        adversarial_weight: float = 0.0,
        physics_weight: float = 0.0,
        pixel_loss: Literal["l1", "mse", "smooth_l1"] = "l1",
        gan_mode: Literal["vanilla", "lsgan"] = "lsgan",
        perceptual_layer: Literal["relu1_2", "relu2_2", "relu3_3", "relu4_3"] = "relu2_2",
    ) -> None:
        super().__init__()

        if pixel_weight < 0:
            raise ValueError(f"pixel_weight must be >= 0, got {pixel_weight}")

        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.physics_weight = physics_weight

        # Pixel loss (always present)
        if pixel_loss == "l1":
            self._pixel: nn.Module = nn.L1Loss()
        elif pixel_loss == "mse":
            self._pixel = nn.MSELoss()
        elif pixel_loss == "smooth_l1":
            self._pixel = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown pixel_loss: {pixel_loss!r}")

        # Perceptual loss (lazy — only built if weight > 0)
        self._perceptual: PerceptualLoss | None = None
        if perceptual_weight > 0:
            self._perceptual = PerceptualLoss(layer=perceptual_layer)

        # Adversarial loss (lazy — only built if weight > 0)
        self._adversarial: GeneratorAdversarialLoss | None = None
        if adversarial_weight > 0:
            self._adversarial = GeneratorAdversarialLoss(mode=gan_mode)

        # Physics loss (lazy — only built if weight > 0)
        self._physics: PhysicsLoss | None = None
        if physics_weight > 0:
            self._physics = PhysicsLoss()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        disc_output: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the combined loss.

        Args:
            prediction: Generated image, shape ``(B, C, H, W)``.
            target: Ground-truth image, same shape as prediction.
            disc_output: Discriminator output for the generated image.
                Required when ``adversarial_weight > 0``.

        Returns:
            Dict with keys:
                - ``"total"``: scalar combined loss
                - ``"pixel"``: pixel reconstruction loss
                - ``"perceptual"``: perceptual loss (if enabled)
                - ``"adversarial"``: adversarial loss (if enabled)
                - ``"physics"``: physics loss (if enabled)
        """
        components: dict[str, torch.Tensor] = {}

        # Pixel loss (always)
        pixel_loss = self._pixel(prediction, target)
        components["pixel"] = pixel_loss
        total = self.pixel_weight * pixel_loss

        # Perceptual loss (optional)
        if self._perceptual is not None and self.perceptual_weight > 0:
            perc_loss = self._perceptual(prediction, target)
            components["perceptual"] = perc_loss
            total = total + self.perceptual_weight * perc_loss

        # Adversarial loss (optional)
        if self._adversarial is not None and self.adversarial_weight > 0:
            if disc_output is None:
                raise ValueError(
                    "disc_output is required when adversarial_weight > 0"
                )
            adv_loss = self._adversarial(disc_output)
            components["adversarial"] = adv_loss
            total = total + self.adversarial_weight * adv_loss

        # Physics loss (optional)
        if self._physics is not None and self.physics_weight > 0:
            phys_loss = self._physics(prediction, target)
            components["physics"] = phys_loss
            total = total + self.physics_weight * phys_loss

        components["total"] = total
        return components
