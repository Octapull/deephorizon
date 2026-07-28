"""Unit tests for services.ml.models.pix2pix."""
from __future__ import annotations

import pytest
import torch

from services.ml.models.pix2pix import PatchDiscriminator, Pix2PixGenerator


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------


def test_generator_output_shape_default() -> None:
    """Pix2PixGenerator(1→1) — çıktı (B, 1, H, W) şeklinde."""
    model = Pix2PixGenerator(in_channels=1, out_channels=1)
    model.eval()

    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 1, 64, 64)


def test_generator_output_shape_custom_channels() -> None:
    """Pix2PixGenerator(3→3) — RGB giriş/çıkış desteklenir."""
    model = Pix2PixGenerator(in_channels=3, out_channels=3)
    model.eval()

    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 3, 32, 32)


def test_generator_tanh_output_range() -> None:
    """use_tanh=True → çıktı [-1, 1] aralığında."""
    model = Pix2PixGenerator(in_channels=1, out_channels=1, use_tanh=True)
    model.eval()

    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        y = model(x)

    assert y.min() >= -1.0
    assert y.max() <= 1.0


def test_generator_no_tanh_unbounded() -> None:
    """use_tanh=False → çıktı sınırsız (lineer)."""
    model = Pix2PixGenerator(in_channels=1, out_channels=1, use_tanh=False)
    model.eval()

    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        y = model(x)

    # Lineer çıkış — Tanh yok, bu yüzden [-1, 1] dışına çıkabilir.
    # En azından tensör şekli doğru ve finite olmalı.
    assert y.shape == (1, 1, 32, 32)
    assert torch.isfinite(y).all()


def test_generator_invalid_dropout() -> None:
    """dropout ∈ [0, 1] dışında → ValueError."""
    with pytest.raises(ValueError, match="dropout"):
        Pix2PixGenerator(dropout=1.5)
    with pytest.raises(ValueError, match="dropout"):
        Pix2PixGenerator(dropout=-0.1)


# ---------------------------------------------------------------------------
# Discriminator tests
# ---------------------------------------------------------------------------


def test_discriminator_output_shape() -> None:
    """PatchDiscriminator — çıktı (B, 1, H_out, W_out) patch haritası."""
    model = PatchDiscriminator(in_channels=2)
    model.eval()

    condition = torch.randn(2, 1, 64, 64)
    target = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        y = model(condition, target)

    # 4 katmanlı Conv(stride=2) → 64/16 = 4 (yaklaşık; padding=1 ile)
    assert y.shape[0] == 2
    assert y.shape[1] == 1
    assert y.shape[2] < 64  # downsampled
    assert y.shape[3] < 64


def test_discriminator_sigmoid_range() -> None:
    """use_sigmoid=True → çıktı [0, 1] aralığında."""
    model = PatchDiscriminator(in_channels=2, use_sigmoid=True)
    model.eval()

    condition = torch.randn(1, 1, 32, 32)
    target = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        y = model(condition, target)

    assert y.min() >= 0.0
    assert y.max() <= 1.0


def test_discriminator_spatial_mismatch_raises() -> None:
    """condition ve target farklı boyutta → ValueError."""
    model = PatchDiscriminator(in_channels=2)

    condition = torch.randn(1, 1, 32, 32)
    target = torch.randn(1, 1, 64, 64)

    with pytest.raises(ValueError, match="Spatial size mismatch"):
        model(condition, target)


def test_discriminator_invalid_n_layers() -> None:
    """n_layers < 1 → ValueError."""
    with pytest.raises(ValueError, match="n_layers"):
        PatchDiscriminator(n_layers=0)


# ---------------------------------------------------------------------------
# Parametre sayısı testleri (sanity check)
# ---------------------------------------------------------------------------


def test_generator_param_count_reasonable() -> None:
    """Pix2PixGenerator ~31M parametre civarında (U-Net bazlı)."""
    model = Pix2PixGenerator(in_channels=1, out_channels=1)
    n_params = sum(p.numel() for p in model.parameters())

    # 31M ± 5M makul bir aralık (U-Net 1024 bottleneck ile).
    assert 25_000_000 < n_params < 40_000_000


def test_discriminator_param_count_reasonable() -> None:
    """PatchDiscriminator ~2.7M parametre civarında (C64-C512)."""
    model = PatchDiscriminator(in_channels=2)
    n_params = sum(p.numel() for p in model.parameters())

    # PatchGAN tipik olarak 2-3M parametre.
    assert 1_000_000 < n_params < 5_000_000
