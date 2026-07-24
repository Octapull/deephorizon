"""Unit tests for services.ml.checkpoints.checkpoint."""
from __future__ import annotations

from pathlib import Path

import torch

from services.ml.checkpoints.checkpoint import load_checkpoint, save_checkpoint
from services.ml.models.unet import UNet


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    """save → load → aynı değerler geri gelir."""
    model = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"

    save_checkpoint(model, optimizer, 1, 0.5, 0.4, 30.0, 0.9, path)

    # Yeni model + optimizer ile load
    new_model = UNet()
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    state = load_checkpoint(path, new_model, new_optimizer)

    assert state["epoch"] == 1
    assert state["train_loss"] == 0.5
    assert state["val_loss"] == 0.4
    assert state["psnr"] == 30.0
    assert state["ssim"] == 0.9


def test_checkpoint_creates_parent_dir(tmp_path: Path) -> None:
    """save_checkpoint üst dizini otomatik oluşturur."""
    model = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "nested" / "dir" / "ckpt.pt"

    save_checkpoint(model, optimizer, 1, 0.5, 0.4, 30.0, 0.9, path)
    assert path.exists()


def test_checkpoint_load_without_optimizer(tmp_path: Path) -> None:
    """optimizer=None → sadece model state_dict yüklenir."""
    model = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"

    save_checkpoint(model, optimizer, 5, 0.3, 0.2, 28.0, 0.85, path)

    new_model = UNet()
    state = load_checkpoint(path, new_model, optimizer=None)

    assert state["epoch"] == 5
    assert state["psnr"] == 28.0


def test_checkpoint_backward_compat_missing_psnr(tmp_path: Path) -> None:
    """Eski checkpoint (psnr/ssim yok) → setdefault ile 0.0 döner."""
    model = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "old_ckpt.pt"

    # Eski format — sadece temel alanlar
    torch.save(
        {
            "epoch": 3,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": 0.5,
            "val_loss": 0.4,
        },
        path,
    )

    new_model = UNet()
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    state = load_checkpoint(path, new_model, new_optimizer)

    assert state["epoch"] == 3
    assert state["psnr"] == 0.0  # backward compat
    assert state["ssim"] == 0.0  # backward compat


def test_checkpoint_state_dict_matches(tmp_path: Path) -> None:
    """Model state_dict load sonrası aynı ağırlıkları üretir."""
    model = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"

    # Modeli bir adım eğit (ağırlıklar değişsin)
    x = torch.rand(1, 1, 64, 64)
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    save_checkpoint(model, optimizer, 1, 0.5, 0.4, 30.0, 0.9, path)

    # Yeni model + aynı input → aynı çıktı olmalı
    new_model = UNet()
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    load_checkpoint(path, new_model, new_optimizer)

    new_model.eval()
    model.eval()
    with torch.no_grad():
        out_original = model(x)
        out_loaded = new_model(x)

    assert torch.allclose(out_original, out_loaded, atol=1e-6)
