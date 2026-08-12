"""ESRGAN (Enhanced SRGAN) model family for super-resolution.

This package implements the ESRGAN architecture (Wang et al., 2018) with
the following components:

- ``rrdb``: Residual-in-Residual Dense Block — the basic building block
  of the ESRGAN generator. 3 dense layers with residual scaling (β=0.2).
- ``generator``: ESRGANGenerator — 23 RRDB blocks + sub-pixel upsampling.
  Supports scale=1 (Pix2Pix-compatible), scale=2, and scale=4.
- ``discriminator``: RaDiscriminator — relativistic average PatchGAN
  discriminator for ESRGAN adversarial training.

Reference:
    Wang, X. et al. (2018). "ESRGAN: Enhanced Super-Resolution
    Generative Adversarial Networks." ECCV Workshops.
"""

from __future__ import annotations

from services.ml.models.esrgan.discriminator import RaDiscriminator
from services.ml.models.esrgan.generator import ESRGANGenerator
from services.ml.models.esrgan.rrdb import DenseLayer, RRDB

__all__ = [
    "DenseLayer",
    "ESRGANGenerator",
    "RaDiscriminator",
    "RRDB",
]
