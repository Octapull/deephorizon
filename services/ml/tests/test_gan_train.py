"""Unit tests for services.ml.training.gan_train helper functions.

The full training loop requires Hydra + MLflow + GPU, so we test the
helper functions in isolation:
    - _build_generator
    - _build_discriminator
    - _build_losses
    - _build_optimizers
    - _train_discriminator_step
    - _train_generator_step

These tests use a minimal DictConfig stub to avoid loading the full
Hydra config tree.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from services.ml.losses.combined import CombinedLoss
from services.ml.losses.gan import DiscriminatorAdversarialLoss
from services.ml.models.pix2pix import PatchDiscriminator, Pix2PixGenerator
from services.ml.training.gan_train import (
    _build_discriminator,
    _build_generator,
    _build_losses,
    _build_optimizers,
    _train_discriminator_step,
    _train_generator_step,
)

# ---------------------------------------------------------------------------
# Config stub
# ---------------------------------------------------------------------------


def _make_config(
    in_channels: int = 1,
    out_channels: int = 1,
    dropout: float = 0.5,
    use_tanh: bool = True,
    disc_in_channels: int = 2,
    disc_base_channels: int = 64,
    disc_max_channels: int = 512,
    disc_n_layers: int = 3,
    disc_use_sigmoid: bool = True,
    pixel_weight: float = 100.0,
    perceptual_weight: float = 0.0,
    adversarial_weight: float = 1.0,
    physics_weight: float = 0.0,
    pixel_loss: str = "l1",
    gan_mode: str = "lsgan",
    perceptual_layer: str = "relu2_2",
    learning_rate: float = 1e-4,
    disc_lr_factor: float = 0.5,
) -> SimpleNamespace:
    """Build a minimal config stub mimicking Hydra's DictConfig."""
    return SimpleNamespace(
        model=SimpleNamespace(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            use_tanh=use_tanh,
            discriminator=SimpleNamespace(
                in_channels=disc_in_channels,
                base_channels=disc_base_channels,
                max_channels=disc_max_channels,
                n_layers=disc_n_layers,
                use_sigmoid=disc_use_sigmoid,
            ),
        ),
        loss=SimpleNamespace(
            weights=SimpleNamespace(
                pixel=pixel_weight,
                perceptual=perceptual_weight,
                adversarial=adversarial_weight,
                physics=physics_weight,
            ),
            pixel_loss=pixel_loss,
            gan_mode=gan_mode,
            perceptual_layer=perceptual_layer,
        ),
        training=SimpleNamespace(
            learning_rate=learning_rate,
            optimizer=SimpleNamespace(
                betas=[0.9, 0.999],
                weight_decay=0.0,
            ),
            discriminator_lr_factor=disc_lr_factor,
        ),
    )


# ---------------------------------------------------------------------------
# _build_generator
# ---------------------------------------------------------------------------


def test_build_generator_default() -> None:
    """_build_generator → Pix2PixGenerator on CPU."""
    cfg = _make_config()
    device = torch.device("cpu")
    generator = _build_generator(cfg, device)

    assert isinstance(generator, Pix2PixGenerator)
    assert generator.in_channels == 1
    assert generator.out_channels == 1
    assert generator.dropout == 0.5
    assert generator.use_tanh is True


def test_build_generator_custom_channels() -> None:
    """_build_generator(3→3) → RGB generator."""
    cfg = _make_config(in_channels=3, out_channels=3)
    device = torch.device("cpu")
    generator = _build_generator(cfg, device)

    assert generator.in_channels == 3
    assert generator.out_channels == 3


def test_build_generator_no_tanh() -> None:
    """_build_generator(use_tanh=False) → lineer çıkış."""
    cfg = _make_config(use_tanh=False)
    device = torch.device("cpu")
    generator = _build_generator(cfg, device)

    assert generator.use_tanh is False


# ---------------------------------------------------------------------------
# _build_discriminator
# ---------------------------------------------------------------------------


def test_build_discriminator_default() -> None:
    """_build_discriminator → PatchDiscriminator on CPU."""
    cfg = _make_config()
    device = torch.device("cpu")
    discriminator = _build_discriminator(cfg, device)

    assert isinstance(discriminator, PatchDiscriminator)
    assert discriminator.in_channels == 2
    assert discriminator.base_channels == 64
    assert discriminator.max_channels == 512
    assert discriminator.n_layers == 3
    assert discriminator.use_sigmoid is True


def test_build_discriminator_no_sigmoid() -> None:
    """_build_discriminator(use_sigmoid=False) → LSGAN için uygun."""
    cfg = _make_config(disc_use_sigmoid=False)
    device = torch.device("cpu")
    discriminator = _build_discriminator(cfg, device)

    assert discriminator.use_sigmoid is False


# ---------------------------------------------------------------------------
# _build_losses
# ---------------------------------------------------------------------------


def test_build_losses_default() -> None:
    """_build_losses → CombinedLoss + DiscriminatorAdversarialLoss."""
    cfg = _make_config()
    g_loss, d_loss = _build_losses(cfg)

    assert isinstance(g_loss, CombinedLoss)
    assert isinstance(d_loss, DiscriminatorAdversarialLoss)
    assert g_loss.pixel_weight == 100.0
    assert g_loss.adversarial_weight == 1.0
    assert d_loss.mode == "lsgan"


def test_build_losses_with_physics() -> None:
    """_build_losses(physics_weight > 0) → physics bileşeni aktif."""
    cfg = _make_config(physics_weight=1.0)
    g_loss, _ = _build_losses(cfg)

    assert g_loss.physics_weight == 1.0
    assert g_loss._physics is not None


def test_build_losses_vanilla_mode() -> None:
    """_build_losses(gan_mode='vanilla') → BCE modu."""
    cfg = _make_config(gan_mode="vanilla")
    _, d_loss = _build_losses(cfg)

    assert d_loss.mode == "vanilla"


# ---------------------------------------------------------------------------
# _build_optimizers
# ---------------------------------------------------------------------------


def test_build_optimizers_lr_factor() -> None:
    """_build_optimizers → D lr = G lr × 0.5."""
    cfg = _make_config(learning_rate=1e-3, disc_lr_factor=0.5)
    generator = Pix2PixGenerator()
    discriminator = PatchDiscriminator()

    g_opt, d_opt = _build_optimizers(cfg, generator, discriminator)

    assert g_opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert d_opt.param_groups[0]["lr"] == pytest.approx(5e-4)


def test_build_optimizers_adam() -> None:
    """_build_optimizers → Adam optimizer."""
    cfg = _make_config()
    generator = Pix2PixGenerator()
    discriminator = PatchDiscriminator()

    g_opt, d_opt = _build_optimizers(cfg, generator, discriminator)

    assert isinstance(g_opt, torch.optim.Adam)
    assert isinstance(d_opt, torch.optim.Adam)


# ---------------------------------------------------------------------------
# _train_discriminator_step
# ---------------------------------------------------------------------------


def test_train_discriminator_step_returns_float() -> None:
    """_train_discriminator_step → float loss döner."""
    discriminator = PatchDiscriminator(in_channels=2, base_channels=4, n_layers=2)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    d_loss_fn = DiscriminatorAdversarialLoss(mode="lsgan")

    condition = torch.rand(2, 1, 16, 16)
    real_target = torch.rand(2, 1, 16, 16)
    fake_target = torch.rand(2, 1, 16, 16)

    loss = _train_discriminator_step(
        discriminator=discriminator,
        d_optimizer=d_optimizer,
        d_loss_fn=d_loss_fn,
        condition=condition,
        real_target=real_target,
        fake_target=fake_target,
        use_amp=False,
        amp_dtype=torch.float32,
        scaler=None,
    )

    assert isinstance(loss, float)
    assert loss >= 0.0


def test_train_discriminator_step_updates_params() -> None:
    """_train_discriminator_step → D parametreleri değişir."""
    discriminator = PatchDiscriminator(in_channels=2, base_channels=4, n_layers=2)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    d_loss_fn = DiscriminatorAdversarialLoss(mode="lsgan")

    initial_params = [p.clone() for p in discriminator.parameters()]

    condition = torch.rand(2, 1, 16, 16)
    real_target = torch.rand(2, 1, 16, 16)
    fake_target = torch.rand(2, 1, 16, 16)

    _train_discriminator_step(
        discriminator=discriminator,
        d_optimizer=d_optimizer,
        d_loss_fn=d_loss_fn,
        condition=condition,
        real_target=real_target,
        fake_target=fake_target,
        use_amp=False,
        amp_dtype=torch.float32,
        scaler=None,
    )

    # En az bir parametre değişmeli
    changed = any(
        not torch.equal(initial, current)
        for initial, current in zip(initial_params, discriminator.parameters())
    )
    assert changed


# ---------------------------------------------------------------------------
# _train_generator_step
# ---------------------------------------------------------------------------


def test_train_generator_step_returns_components() -> None:
    """_train_generator_step → dict with 'total', 'pixel', 'adversarial'."""
    generator = Pix2PixGenerator(in_channels=1, out_channels=1)
    discriminator = PatchDiscriminator(in_channels=2, base_channels=4, n_layers=2)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
    g_loss_fn = CombinedLoss(
        pixel_weight=100.0,
        adversarial_weight=1.0,
        gan_mode="lsgan",
    )

    condition = torch.rand(2, 1, 16, 16)
    real_target = torch.rand(2, 1, 16, 16)

    components = _train_generator_step(
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        g_loss_fn=g_loss_fn,
        condition=condition,
        real_target=real_target,
        use_amp=False,
        amp_dtype=torch.float32,
        scaler=None,
    )

    assert "total" in components
    assert "pixel" in components
    assert "adversarial" in components
    assert components["total"] >= 0.0


def test_train_generator_step_updates_params() -> None:
    """_train_generator_step → G parametreleri değişir."""
    generator = Pix2PixGenerator(in_channels=1, out_channels=1)
    discriminator = PatchDiscriminator(in_channels=2, base_channels=4, n_layers=2)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
    g_loss_fn = CombinedLoss(
        pixel_weight=100.0,
        adversarial_weight=1.0,
        gan_mode="lsgan",
    )

    initial_params = [p.clone() for p in generator.parameters()]

    condition = torch.rand(2, 1, 16, 16)
    real_target = torch.rand(2, 1, 16, 16)

    _train_generator_step(
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        g_loss_fn=g_loss_fn,
        condition=condition,
        real_target=real_target,
        use_amp=False,
        amp_dtype=torch.float32,
        scaler=None,
    )

    changed = any(
        not torch.equal(initial, current)
        for initial, current in zip(initial_params, generator.parameters())
    )
    assert changed


def test_train_generator_step_with_physics() -> None:
    """_train_generator_step(physics_weight > 0) → 'physics' bileşeni döner."""
    generator = Pix2PixGenerator(in_channels=1, out_channels=1)
    discriminator = PatchDiscriminator(in_channels=2)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
    g_loss_fn = CombinedLoss(
        pixel_weight=100.0,
        adversarial_weight=1.0,
        physics_weight=1.0,
        gan_mode="lsgan",
    )

    condition = torch.rand(2, 1, 32, 32)
    real_target = torch.rand(2, 1, 32, 32)

    components = _train_generator_step(
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        g_loss_fn=g_loss_fn,
        condition=condition,
        real_target=real_target,
        use_amp=False,
        amp_dtype=torch.float32,
        scaler=None,
    )

    assert "physics" in components
    assert components["physics"] >= 0.0
