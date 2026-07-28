"""Unit tests for services.ml.losses.gan (adversarial losses)."""
from __future__ import annotations

import pytest
import torch

from services.ml.losses.gan import (
    DiscriminatorAdversarialLoss,
    GeneratorAdversarialLoss,
)


# ---------------------------------------------------------------------------
# Generator adversarial loss
# ---------------------------------------------------------------------------


def test_generator_vanilla_returns_scalar() -> None:
    """GeneratorAdversarialLoss(vanilla) → scalar tensor."""
    loss_fn = GeneratorAdversarialLoss(mode="vanilla")
    disc_output = torch.rand(2, 1, 8, 8)  # sigmoid output in [0, 1]
    value = loss_fn(disc_output)
    assert value.dim() == 0
    assert value.item() >= 0.0


def test_generator_lsgan_returns_scalar() -> None:
    """GeneratorAdversarialLoss(lsgan) → scalar tensor."""
    loss_fn = GeneratorAdversarialLoss(mode="lsgan")
    disc_output = torch.rand(2, 1, 8, 8)
    value = loss_fn(disc_output)
    assert value.dim() == 0
    assert value.item() >= 0.0


def test_generator_lsgan_zero_when_perfect() -> None:
    """LSGAN: D(G(z))=1 → loss = 0."""
    loss_fn = GeneratorAdversarialLoss(mode="lsgan")
    disc_output = torch.ones(2, 1, 8, 8)
    value = loss_fn(disc_output)
    assert value.item() == pytest.approx(0.0, abs=1e-6)


def test_generator_invalid_mode() -> None:
    """mode='unknown' → ValueError."""
    with pytest.raises(ValueError, match="mode"):
        GeneratorAdversarialLoss(mode="wgan")


# ---------------------------------------------------------------------------
# Discriminator adversarial loss
# ---------------------------------------------------------------------------


def test_discriminator_vanilla_returns_scalar() -> None:
    """DiscriminatorAdversarialLoss(vanilla) → scalar tensor."""
    loss_fn = DiscriminatorAdversarialLoss(mode="vanilla")
    disc_real = torch.rand(2, 1, 8, 8)
    disc_fake = torch.rand(2, 1, 8, 8)
    value = loss_fn(disc_real, disc_fake)
    assert value.dim() == 0
    assert value.item() >= 0.0


def test_discriminator_lsgan_returns_scalar() -> None:
    """DiscriminatorAdversarialLoss(lsgan) → scalar tensor."""
    loss_fn = DiscriminatorAdversarialLoss(mode="lsgan")
    disc_real = torch.rand(2, 1, 8, 8)
    disc_fake = torch.rand(2, 1, 8, 8)
    value = loss_fn(disc_real, disc_fake)
    assert value.dim() == 0
    assert value.item() >= 0.0


def test_discriminator_lsgan_zero_when_perfect() -> None:
    """LSGAN: D(real)=1, D(fake)=0 → loss = 0."""
    loss_fn = DiscriminatorAdversarialLoss(mode="lsgan")
    disc_real = torch.ones(2, 1, 8, 8)
    disc_fake = torch.zeros(2, 1, 8, 8)
    value = loss_fn(disc_real, disc_fake)
    assert value.item() == pytest.approx(0.0, abs=1e-6)


def test_discriminator_invalid_mode() -> None:
    """mode='unknown' → ValueError."""
    with pytest.raises(ValueError, match="mode"):
        DiscriminatorAdversarialLoss(mode="hinge")


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


def test_factory_gan_g() -> None:
    """get_loss('gan_g') → GeneratorAdversarialLoss."""
    from services.ml.losses.loss import get_loss

    loss = get_loss("gan_g", mode="lsgan")
    assert isinstance(loss, GeneratorAdversarialLoss)
    assert loss.mode == "lsgan"


def test_factory_gan_d() -> None:
    """get_loss('gan_d') → DiscriminatorAdversarialLoss."""
    from services.ml.losses.loss import get_loss

    loss = get_loss("gan_d", mode="vanilla")
    assert isinstance(loss, DiscriminatorAdversarialLoss)
    assert loss.mode == "vanilla"
