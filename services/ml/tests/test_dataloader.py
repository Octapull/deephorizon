"""Unit tests for services.ml.data.dataloader."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from services.ml.data.dataloader import create_dataloader, create_train_val_loaders


@pytest.fixture
def dataset_10(tmp_path: Path) -> Path:
    """10 çift .npy içeren geçici dataset."""
    clean_dir = tmp_path / "clean"
    degraded_dir = tmp_path / "degraded"
    clean_dir.mkdir(parents=True)
    degraded_dir.mkdir(parents=True)

    for i in range(10):
        np.save(clean_dir / f"img_{i:03d}.npy", np.random.rand(32, 32).astype(np.float32))
        np.save(degraded_dir / f"img_{i:03d}.npy", np.random.rand(32, 32).astype(np.float32))

    return tmp_path


def test_create_dataloader_returns_dataloader(dataset_10: Path) -> None:
    """create_dataloader → DataLoader instance."""
    loader = create_dataloader(
        root_dir=dataset_10,
        batch_size=2,
        num_workers=0,
    )
    assert isinstance(loader, torch.utils.data.DataLoader)


def test_create_dataloader_batch_shape(dataset_10: Path) -> None:
    """Batch shape = (batch_size, 1, H, W)."""
    loader = create_dataloader(
        root_dir=dataset_10,
        batch_size=2,
        num_workers=0,
    )
    batch = next(iter(loader))
    degraded, clean = batch
    assert degraded.shape[0] == 2
    assert clean.shape[0] == 2
    assert degraded.shape[1] == 1
    assert clean.shape[1] == 1


def test_create_train_val_loaders_split_ratio(dataset_10: Path) -> None:
    """val_ratio=0.2 → 8 train + 2 val."""
    train_loader, val_loader = create_train_val_loaders(
        root_dir=dataset_10,
        batch_size=2,
        val_ratio=0.2,
        num_workers=0,
    )
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    assert train_size == 8
    assert val_size == 2


def test_create_train_val_loaders_invalid_ratio(dataset_10: Path) -> None:
    """val_ratio=0 veya 1 → ValueError."""
    with pytest.raises(ValueError, match="val_ratio"):
        create_train_val_loaders(
            root_dir=dataset_10,
            batch_size=2,
            val_ratio=0.0,
            num_workers=0,
        )


def test_create_train_val_loaders_too_few_samples(tmp_path: Path) -> None:
    """1 örnek → ValueError (en az 2 gerekli)."""
    (tmp_path / "clean").mkdir()
    (tmp_path / "degraded").mkdir()
    np.save(tmp_path / "clean" / "img_000.npy", np.random.rand(8, 8).astype(np.float32))
    np.save(tmp_path / "degraded" / "img_000.npy", np.random.rand(8, 8).astype(np.float32))

    with pytest.raises(ValueError, match="At least 2 samples"):
        create_train_val_loaders(
            root_dir=tmp_path,
            batch_size=1,
            val_ratio=0.5,
            num_workers=0,
        )


def test_create_train_val_loaders_val_no_shuffle(dataset_10: Path) -> None:
    """Validation loader shuffle=False (deterministic ölçüm)."""
    _, val_loader = create_train_val_loaders(
        root_dir=dataset_10,
        batch_size=2,
        val_ratio=0.2,
        num_workers=0,
    )
    # İlk batch'i iki kez al → aynı olmalı (shuffle=False)
    batch1 = next(iter(val_loader))
    batch2 = next(iter(val_loader))
    assert torch.equal(batch1[0], batch2[0])


def test_create_train_val_loaders_with_split(dataset_10: Path) -> None:
    """split='medium' → root_dir/medium/clean + root_dir/medium/degraded."""
    medium_dir = dataset_10 / "medium"
    (medium_dir / "clean").mkdir(parents=True)
    (medium_dir / "degraded").mkdir(parents=True)
    for i in range(6):
        np.save(medium_dir / "clean" / f"img_{i:03d}.npy", np.random.rand(16, 16).astype(np.float32))
        np.save(medium_dir / "degraded" / f"img_{i:03d}.npy", np.random.rand(16, 16).astype(np.float32))

    train_loader, val_loader = create_train_val_loaders(
        root_dir=dataset_10,
        batch_size=2,
        val_ratio=0.5,
        num_workers=0,
        split="medium",
    )
    assert len(train_loader.dataset) == 3
    assert len(val_loader.dataset) == 3
