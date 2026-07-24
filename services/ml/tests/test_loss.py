"""Unit tests for services.ml.losses.loss factory."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from services.ml.losses.loss import _SUPPORTED_LOSSES, get_loss


def test_get_loss_default_is_mse() -> None:
    """Default loss = MSE."""
    loss = get_loss()
    assert isinstance(loss, nn.MSELoss)


def test_get_loss_mse() -> None:
    """name='mse' → MSELoss."""
    loss = get_loss("mse")
    assert isinstance(loss, nn.MSELoss)


def test_get_loss_l1() -> None:
    """name='l1' → L1Loss."""
    loss = get_loss("l1")
    assert isinstance(loss, nn.L1Loss)


def test_get_loss_smooth_l1() -> None:
    """name='smooth_l1' → SmoothL1Loss."""
    loss = get_loss("smooth_l1")
    assert isinstance(loss, nn.SmoothL1Loss)


def test_get_loss_smooth_l1_with_beta() -> None:
    """kwargs forward — beta parametresi SmoothL1Loss'a geçer."""
    loss = get_loss("smooth_l1", beta=0.5)
    assert isinstance(loss, nn.SmoothL1Loss)
    assert loss.beta == 0.5


def test_get_loss_mse_with_reduction() -> None:
    """kwargs forward — reduction parametresi MSELoss'a geçer."""
    loss = get_loss("mse", reduction="sum")
    assert isinstance(loss, nn.MSELoss)
    assert loss.reduction == "sum"


def test_get_loss_unknown_raises() -> None:
    """Bilinmeyen loss → ValueError."""
    with pytest.raises(ValueError, match="Unknown loss"):
        get_loss("unknown")


def test_get_loss_physics_not_yet_supported() -> None:
    """Faz 2'de eklenecek physics loss şimdilik ValueError."""
    with pytest.raises(ValueError, match="Unknown loss"):
        get_loss("physics")


def test_supported_losses_is_frozenset() -> None:
    """_SUPPORTED_LOSSES immutable (frozenset)."""
    assert isinstance(_SUPPORTED_LOSSES, frozenset)
    assert "mse" in _SUPPORTED_LOSSES
    assert "l1" in _SUPPORTED_LOSSES
    assert "smooth_l1" in _SUPPORTED_LOSSES


def test_loss_forward_computes_scalar() -> None:
    """Loss forward → scalar tensor."""
    loss_fn = get_loss("l1")
    pred = torch.rand(2, 1, 8, 8)
    target = torch.rand(2, 1, 8, 8)
    value = loss_fn(pred, target)
    assert value.dim() == 0  # scalar
    assert value.item() >= 0.0
