"""Unit tests for services.ml.models.esrgan."""

from __future__ import annotations

import pytest
import torch

from services.ml.models.esrgan import DenseLayer, ESRGANGenerator, RaDiscriminator, RRDB

# ---------------------------------------------------------------------------
# DenseLayer tests
# ---------------------------------------------------------------------------


def test_dense_layer_output_shape() -> None:
    """DenseLayer → (B, growth_rate, H, W) çıktı."""
    layer = DenseLayer(in_channels=32, growth_rate=32)
    layer.eval()

    x = torch.randn(2, 32, 16, 16)
    with torch.no_grad():
        y = layer(x)

    assert y.shape == (2, 32, 16, 16)


def test_dense_layer_growth_rate() -> None:
    """DenseLayer çıktı kanal sayısı = growth_rate."""
    layer = DenseLayer(in_channels=64, growth_rate=16)
    layer.eval()

    x = torch.randn(1, 64, 8, 8)
    with torch.no_grad():
        y = layer(x)

    assert y.shape[1] == 16


# ---------------------------------------------------------------------------
# RRDB tests
# ---------------------------------------------------------------------------


def test_rrdb_output_shape() -> None:
    """RRDB → (B, features, H, W) çıktı (giriş ile aynı shape)."""
    block = RRDB(features=32, growth_rate=32, num_dense_layers=3)
    block.eval()

    x = torch.randn(2, 32, 16, 16)
    with torch.no_grad():
        y = block(x)

    assert y.shape == x.shape


def test_rrdb_residual_scaling() -> None:
    """RRDB residual_scale=0 → çıktı = giriş (identity)."""
    block = RRDB(features=32, growth_rate=32, num_dense_layers=3, residual_scale=0.0)
    block.eval()

    x = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        y = block(x)

    assert torch.allclose(y, x, atol=1e-6)


def test_rrdb_invalid_num_dense_layers() -> None:
    """num_dense_layers < 1 → ValueError."""
    with pytest.raises(ValueError, match="num_dense_layers"):
        RRDB(num_dense_layers=0)


def test_rrdb_invalid_residual_scale() -> None:
    """residual_scale ∈ [0, 1] dışında → ValueError."""
    with pytest.raises(ValueError, match="residual_scale"):
        RRDB(residual_scale=1.5)
    with pytest.raises(ValueError, match="residual_scale"):
        RRDB(residual_scale=-0.1)


def test_rrdb_gradient_flow() -> None:
    """RRDB backward pass → gradient akışı var."""
    block = RRDB(features=32, growth_rate=32, num_dense_layers=3)
    x = torch.randn(1, 32, 8, 8, requires_grad=True)
    y = block(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# ESRGANGenerator tests
# ---------------------------------------------------------------------------


def test_esrgan_generator_scale_1() -> None:
    """ESRGANGenerator(scale=1) → giriş ile aynı spatial boyut."""
    model = ESRGANGenerator(
        in_channels=1, out_channels=1, num_rrdb=2, features=32, scale=1
    )
    model.eval()

    x = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 1, 32, 32)


def test_esrgan_generator_scale_2() -> None:
    """ESRGANGenerator(scale=2) → 2x büyütme."""
    model = ESRGANGenerator(
        in_channels=1, out_channels=1, num_rrdb=2, features=32, scale=2
    )
    model.eval()

    x = torch.randn(1, 1, 16, 16)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 1, 32, 32)


def test_esrgan_generator_scale_4() -> None:
    """ESRGANGenerator(scale=4) → 4x büyütme."""
    model = ESRGANGenerator(
        in_channels=1, out_channels=1, num_rrdb=2, features=32, scale=4
    )
    model.eval()

    x = torch.randn(1, 1, 8, 8)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 1, 32, 32)


def test_esrgan_generator_tanh_range() -> None:
    """use_tanh=True → çıktı [-1, 1] aralığında."""
    model = ESRGANGenerator(
        in_channels=1, out_channels=1, num_rrdb=2, features=32, scale=1, use_tanh=True
    )
    model.eval()

    x = torch.randn(1, 1, 16, 16)
    with torch.no_grad():
        y = model(x)

    assert y.min() >= -1.0
    assert y.max() <= 1.0


def test_esrgan_generator_invalid_scale() -> None:
    """scale ∈ {1, 2, 4} dışında → ValueError."""
    with pytest.raises(ValueError, match="scale"):
        ESRGANGenerator(scale=3)


def test_esrgan_generator_invalid_num_rrdb() -> None:
    """num_rrdb < 1 → ValueError."""
    with pytest.raises(ValueError, match="num_rrdb"):
        ESRGANGenerator(num_rrdb=0)


def test_esrgan_generator_param_count() -> None:
    """ESRGANGenerator ~9.4M parametre (23 RRDB, features=64, scale=1)."""
    model = ESRGANGenerator(
        in_channels=1, out_channels=1, num_rrdb=23, features=64, scale=1
    )
    n_params = sum(p.numel() for p in model.parameters())

    # 23 RRDB × ~400K + head/tail ≈ 9.4M (scale=1, grayscale)
    assert 8_000_000 < n_params < 11_000_000


# ---------------------------------------------------------------------------
# RaDiscriminator tests
# ---------------------------------------------------------------------------


def test_ra_discriminator_output_shape() -> None:
    """RaDiscriminator → (B, 1, H_out, W_out) patch haritası."""
    model = RaDiscriminator(in_channels=2, base_channels=32, n_layers=3)
    model.eval()

    condition = torch.randn(2, 1, 32, 32)
    target = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        y = model(condition, target)

    assert y.shape[0] == 2
    assert y.shape[1] == 1
    assert y.shape[2] < 32  # downsampled
    assert y.shape[3] < 32


def test_ra_discriminator_no_sigmoid() -> None:
    """RaDiscriminator → sigmoid yok, çıktı sınırsız (logit)."""
    model = RaDiscriminator(in_channels=2, base_channels=32, n_layers=3)
    model.eval()

    condition = torch.randn(1, 1, 16, 16)
    target = torch.randn(1, 1, 16, 16)
    with torch.no_grad():
        y = model(condition, target)

    # Sigmoid olmadığı için çıktı [0, 1] dışına çıkabilir
    assert torch.isfinite(y).all()


def test_ra_discriminator_spatial_mismatch() -> None:
    """condition ve target farklı boyutta → ValueError."""
    model = RaDiscriminator(in_channels=2, base_channels=32, n_layers=3)

    condition = torch.randn(1, 1, 16, 16)
    target = torch.randn(1, 1, 32, 32)

    with pytest.raises(ValueError, match="Spatial size mismatch"):
        model(condition, target)


def test_ra_discriminator_invalid_n_layers() -> None:
    """n_layers < 1 → ValueError."""
    with pytest.raises(ValueError, match="n_layers"):
        RaDiscriminator(n_layers=0)


def test_ra_discriminator_relativistic_logits() -> None:
    """relativistic_logits → D_real_rel ve D_fake_rel döner."""
    model = RaDiscriminator(in_channels=2, base_channels=32, n_layers=3)
    model.eval()

    condition = torch.randn(2, 1, 16, 16)
    target = torch.randn(2, 1, 16, 16)
    with torch.no_grad():
        disc_real = model(condition, target)
        disc_fake = model(condition, torch.randn_like(target))
        D_real_rel, D_fake_rel = model.relativistic_logits(disc_real, disc_fake)

    assert D_real_rel.shape == disc_real.shape
    assert D_fake_rel.shape == disc_fake.shape
    # Relativistic logit: D_real_rel = disc_real - mean(disc_fake)
    expected = disc_real - disc_fake.mean(dim=0, keepdim=True)
    assert torch.allclose(D_real_rel, expected, atol=1e-6)


def test_ra_discriminator_param_count() -> None:
    """RaDiscriminator ~7M parametre (5 katman, C64-C512, in_channels=2)."""
    model = RaDiscriminator(in_channels=2, base_channels=64, n_layers=5)
    n_params = sum(p.numel() for p in model.parameters())

    # 5-layer PatchGAN with C64→C512 channel doubling ≈ 7M
    assert 5_000_000 < n_params < 9_000_000
