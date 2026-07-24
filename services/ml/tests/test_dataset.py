"""Unit tests for services.ml.data.dataset.BlackHoleDataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from services.ml.data.dataset import BlackHoleDataset


@pytest.fixture
def local_dataset_dir(tmp_path: Path) -> Path:
    """Geçici yerel dataset dizini — 4 çift clean/degraded .npy."""
    clean_dir = tmp_path / "clean"
    degraded_dir = tmp_path / "degraded"
    clean_dir.mkdir(parents=True)
    degraded_dir.mkdir(parents=True)

    for i in range(4):
        np.save(clean_dir / f"img_{i:03d}.npy", np.random.rand(64, 64).astype(np.float32))
        np.save(degraded_dir / f"img_{i:03d}.npy", np.random.rand(64, 64).astype(np.float32))

    return tmp_path


def test_dataset_len(local_dataset_dir: Path) -> None:
    """len(dataset) = dosya sayısı."""
    ds = BlackHoleDataset(root_dir=local_dataset_dir, use_minio=False)
    assert len(ds) == 4


def test_dataset_getitem_shape(local_dataset_dir: Path) -> None:
    """__getitem__ → (degraded, clean) tuple, shape (1, 1, H, W)."""
    ds = BlackHoleDataset(root_dir=local_dataset_dir, use_minio=False)
    degraded, clean = ds[0]
    assert isinstance(degraded, torch.Tensor)
    assert isinstance(clean, torch.Tensor)
    assert degraded.shape == (1, 1, 64, 64)
    assert clean.shape == (1, 1, 64, 64)


def test_dataset_getitem_range(local_dataset_dir: Path) -> None:
    """Pixel değerleri [0, 1] aralığında (float32)."""
    ds = BlackHoleDataset(root_dir=local_dataset_dir, use_minio=False)
    degraded, clean = ds[0]
    assert degraded.dtype == torch.float32
    assert clean.dtype == torch.float32
    assert degraded.min() >= 0.0
    assert degraded.max() <= 1.0


def test_dataset_split_subdir(local_dataset_dir: Path) -> None:
    """split='medium' → root_dir/medium/clean + root_dir/medium/degraded."""
    medium_dir = local_dataset_dir / "medium"
    (medium_dir / "clean").mkdir(parents=True)
    (medium_dir / "degraded").mkdir(parents=True)
    for i in range(2):
        np.save(medium_dir / "clean" / f"img_{i:03d}.npy", np.random.rand(32, 32).astype(np.float32))
        np.save(medium_dir / "degraded" / f"img_{i:03d}.npy", np.random.rand(32, 32).astype(np.float32))

    ds = BlackHoleDataset(root_dir=local_dataset_dir, use_minio=False, split="medium")
    assert len(ds) == 2
    degraded, clean = ds[0]
    assert degraded.shape == (1, 1, 32, 32)


def test_dataset_augment_changes_shape(local_dataset_dir: Path) -> None:
    """augment=True + crop_size=32 → 64x64'den 32x32'ye crop."""
    ds = BlackHoleDataset(
        root_dir=local_dataset_dir,
        use_minio=False,
        augment=True,
        crop_size=32,
    )
    degraded, clean = ds[0]
    assert degraded.shape == (1, 1, 32, 32)
    assert clean.shape == (1, 1, 32, 32)


def test_dataset_no_augment_keeps_shape(local_dataset_dir: Path) -> None:
    """augment=False → orijinal shape korunur."""
    ds = BlackHoleDataset(root_dir=local_dataset_dir, use_minio=False, augment=False)
    degraded, clean = ds[0]
    assert degraded.shape == (1, 1, 64, 64)


def test_dataset_empty_dir(tmp_path: Path) -> None:
    """Boş dizin → len = 0."""
    (tmp_path / "clean").mkdir()
    (tmp_path / "degraded").mkdir()
    ds = BlackHoleDataset(root_dir=tmp_path, use_minio=False)
    assert len(ds) == 0
