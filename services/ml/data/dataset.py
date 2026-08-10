from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from services.ml.minio_loader import list_files_in_minio, load_npy_from_minio


class BlackHoleDataset(Dataset):
    """Paired clean/degraded black-hole images from MinIO or local storage."""

    def __init__(
        self,
        root_dir,
        use_minio=False,
        bucket_name="datasets",
        minio_prefix="training-512/v1",
        augment=False,
        crop_size=256,
        split=None,
        max_samples=None,
    ):
        self.root_dir = Path(root_dir)
        self.use_minio = use_minio
        self.bucket_name = bucket_name
        self.minio_prefix = minio_prefix.rstrip("/")
        self.augment = augment
        self.crop_size = crop_size
        self.split = split

        if not self.use_minio:
            base = self.root_dir / split if split else self.root_dir
            self.clean_files = sorted((base / "clean").glob("*.npy"))
            self.degraded_files = sorted((base / "degraded").glob("*.npy"))
        else:
            clean_path = (
                f"{self.minio_prefix}/{split}/clean/"
                if split
                else f"{self.minio_prefix}/clean/"
            )
            degraded_path = (
                f"{self.minio_prefix}/{split}/degraded/"
                if split
                else f"{self.minio_prefix}/degraded/"
            )
            self.clean_files = list_files_in_minio(self.bucket_name, clean_path)
            self.degraded_files = list_files_in_minio(self.bucket_name, degraded_path)

        clean_names = [Path(path).name for path in self.clean_files]
        degraded_names = [Path(path).name for path in self.degraded_files]
        if clean_names != degraded_names:
            raise ValueError(
                "Clean/degraded dataset keys do not match; refusing mispaired training data."
            )

        if max_samples is not None:
            limit = int(max_samples)
            if limit < 2:
                raise ValueError("max_samples must be at least 2")
            self.clean_files = self.clean_files[:limit]
            self.degraded_files = self.degraded_files[:limit]

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, index):
        if not self.use_minio:
            clean_data = np.load(self.clean_files[index], allow_pickle=False)
            degraded_data = np.load(self.degraded_files[index], allow_pickle=False)
        else:
            clean_data = load_npy_from_minio(self.bucket_name, self.clean_files[index])
            degraded_data = load_npy_from_minio(
                self.bucket_name, self.degraded_files[index]
            )

        clean = torch.from_numpy(clean_data).unsqueeze(0)
        degraded = torch.from_numpy(degraded_data).unsqueeze(0)

        if self.augment:
            clean, degraded = self._augment(clean, degraded)

        return degraded, clean

    def _augment(self, clean, degraded):
        """Apply identical random transforms to a clean/degraded pair."""
        if torch.rand(()).item() < 0.5:
            degraded = torch.flip(degraded, dims=[-1])
            clean = torch.flip(clean, dims=[-1])

        if torch.rand(()).item() < 0.5:
            degraded = torch.flip(degraded, dims=[-2])
            clean = torch.flip(clean, dims=[-2])

        k = int(torch.randint(0, 4, ()).item())
        if k > 0:
            degraded = torch.rot90(degraded, k=k, dims=[-2, -1])
            clean = torch.rot90(clean, k=k, dims=[-2, -1])

        _, height, width = degraded.shape
        crop = int(self.crop_size)
        if height >= crop and width >= crop:
            top = int(torch.randint(0, height - crop + 1, ()).item())
            left = int(torch.randint(0, width - crop + 1, ()).item())
            degraded = degraded[..., top : top + crop, left : left + crop]
            clean = clean[..., top : top + crop, left : left + crop]

        return clean, degraded
