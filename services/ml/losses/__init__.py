"""Loss functions for DeepHorizon training.

Modules:
    loss: Factory for simple pixel losses (mse, l1, smooth_l1).
    perceptual: VGG19-based perceptual loss.
    gan: Adversarial losses (vanilla BCE, LSGAN) for generator/discriminator.
    physics: Physics-informed loss (flux, ring, asymmetry).
    combined: Weighted sum of pixel + perceptual + adversarial + physics.
"""
from __future__ import annotations

from services.ml.losses.combined import CombinedLoss
from services.ml.losses.gan import (
    DiscriminatorAdversarialLoss,
    GeneratorAdversarialLoss,
)
from services.ml.losses.loss import get_loss
from services.ml.losses.perceptual import PerceptualLoss
from services.ml.losses.physics import PhysicsLoss

__all__ = [
    "CombinedLoss",
    "DiscriminatorAdversarialLoss",
    "GeneratorAdversarialLoss",
    "PerceptualLoss",
    "PhysicsLoss",
    "get_loss",
]
