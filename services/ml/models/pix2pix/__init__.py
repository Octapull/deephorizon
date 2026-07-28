"""Pix2Pix conditional GAN models for image-to-image translation.

This package wraps the deterministic U-Net as a Pix2Pix generator and
provides a 70x70 PatchGAN discriminator (Isola et al., 2017).

Modules:
    generator: Pix2PixGenerator — U-Net wrapper with optional dropout.
    discriminator: PatchDiscriminator — Markovian PatchGAN classifier.
"""
from __future__ import annotations

from services.ml.models.pix2pix.discriminator import PatchDiscriminator
from services.ml.models.pix2pix.generator import Pix2PixGenerator

__all__ = ["PatchDiscriminator", "Pix2PixGenerator"]
