"""Unit tests for services.ml.losses.combined (CombinedLoss)."""
from __future__ import annotations

import pytest
import torch

from services.ml.losses.combined import CombinedLoss


# ---------------------------------------------------------------------------
# Pixel-only mode (default)
# ---------------------------------------------------------------------------


def test_combined_pixel_only() -> None:
    """Sadece pixel weight → sadece 'pixel' ve 'total' bileşenleri."""
    loss_fn = CombinedLoss(pixel_weight=100.0)
    pred = torch.rand(2, 1, 16, 16)
    target = torch.rand(2, 1, 16, 16)

    components = loss_fn(pred, target)

    assert "total" in components
    assert "pixel" in components
    assert "perceptual" not in components
    assert "adversarial" not in components
    assert "physics" not in components
    assert components["total"].dim() == 0
    assert components["total"].item() >= 0.0


def test_combined_pixel_weight_scaling() -> None:
    """pixel_weight=100 → total = 100 * pixel."""
    loss_fn = CombinedLoss(pixel_weight=100.0)
    pred = torch.rand(2, 1, 16, 16)
    target = torch.rand(2, 1, 16, 16)

    components = loss_fn(pred, target)
    expected = 100.0 * components["pixel"].item()
    assert components["total"].item() == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Adversarial mode
# ---------------------------------------------------------------------------


def test_combined_with_adversarial() -> None:
    """adversarial_weight > 0 → 'adversarial' bileşeni eklenir."""
    loss_fn = CombinedLoss(
        pixel_weight=100.0,
        adversarial_weight=1.0,
        gan_mode="lsgan",
    )
    pred = torch.rand(2, 1, 16, 16)
    target = torch.rand(2, 1, 16, 16)
    disc_output = torch.rand(2, 1, 4, 4)  # patch map

    components = loss_fn(pred, target, disc_output=disc_output)

    assert "adversarial" in components
    assert "total" in components
    assert components["adversarial"].dim() == 0


def test_combined_adversarial_requires_disc_output() -> None:
    """adversarial_weight > 0 + disc_output=None → ValueError."""
    loss_fn = CombinedLoss(pixel_weight=100.0, adversarial_weight=1.0)
    pred = torch.rand(2, 1, 16, 16)
    target = torch.rand(2, 1, 16, 16)

    with pytest.raises(ValueError, match="disc_output"):
        loss_fn(pred, target, disc_output=None)


# ---------------------------------------------------------------------------
# Physics mode
# ---------------------------------------------------------------------------


def test_combined_with_physics() -> None:
    """physics_weight > 0 → 'physics' bileşeni eklenir."""
    loss_fn = CombinedLoss(pixel_weight=100.0, physics_weight=1.0)
    pred = torch.rand(2, 1, 32, 32)
    target = torch.rand(2, 1, 32, 32)

    components = loss_fn(pred, target)

    assert "physics" in components
    assert "total" in components
    assert components["physics"].dim() == 0
    assert components["physics"].item() >= 0.0


# ---------------------------------------------------------------------------
# Full mode (all components)
# ---------------------------------------------------------------------------


def test_combined_full_mode() -> None:
    """Tüm bileşenler aktif → 5 anahtar (pixel + perceptual + adv + physics + total)."""
    loss_fn = CombinedLoss(
        pixel_weight=100.0,
        perceptual_weight=0.1,
        adversarial_weight=1.0,
        physics_weight=1.0,
        gan_mode="lsgan",
    )
    pred = torch.rand(2, 1, 32, 32)
    target = torch.rand(2, 1, 32, 32)
    disc_output = torch.rand(2, 1, 4, 4)

    components = loss_fn(pred, target, disc_output=disc_output)

    assert "pixel" in components
    assert "perceptual" in components
    assert "adversarial" in components
    assert "physics" in components
    assert "total" in components


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_combined_invalid_pixel_weight() -> None:
    """pixel_weight < 0 → ValueError."""
    with pytest.raises(ValueError, match="pixel_weight"):
        CombinedLoss(pixel_weight=-1.0)


def test_combined_invalid_pixel_loss() -> None:
    """pixel_loss='unknown' → ValueError."""
    with pytest.raises(ValueError, match="pixel_loss"):
        CombinedLoss(pixel_loss="huber")


def test_combined_pixel_loss_types() -> None:
    """pixel_loss parametresi doğru loss sınıfını seçer."""
    loss_l1 = CombinedLoss(pixel_weight=1.0, pixel_loss="l1")
    loss_mse = CombinedLoss(pixel_weight=1.0, pixel_loss="mse")
    loss_sl1 = CombinedLoss(pixel_weight=1.0, pixel_loss="smooth_l1")

    assert isinstance(loss_l1._pixel, torch.nn.L1Loss)
    assert isinstance(loss_mse._pixel, torch.nn.MSELoss)
    assert isinstance(loss_sl1._pixel, torch.nn.SmoothL1Loss)


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


def test_factory_combined() -> None:
    """get_loss('combined') → CombinedLoss."""
    from services.ml.losses.loss import get_loss

    loss = get_loss("combined", pixel_weight=50.0, adversarial_weight=1.0)
    assert isinstance(loss, CombinedLoss)
    assert loss.pixel_weight == 50.0
    assert loss.adversarial_weight == 1.0


def test_factory_physics() -> None:
    """get_loss('physics') → PhysicsLoss."""
    from services.ml.losses.loss import get_loss
    from services.ml.losses.physics import PhysicsLoss

    loss = get_loss("physics", flux_weight=0.5)
    assert isinstance(loss, PhysicsLoss)
    assert loss.flux_weight == 0.5
