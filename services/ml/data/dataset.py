import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from services.ml.minio_loader import load_npy_from_minio, list_files_in_minio


class BlackHoleDataset(Dataset):
    """Black hole image dataset (clean + degraded pairs).

    Args:
        root_dir: Local kök dizin (use_minio=False ise kullanılır).
        use_minio: True ise MinIO/S3'ten okur, False ise yerel dosya sisteminden.
        bucket_name: MinIO bucket adı.
        minio_prefix: MinIO prefix (clean/ ve degraded/ altında .npy dosyaları).
        augment: True ise random flip + 90° rotation + random crop uygulanır.
        crop_size: Random crop boyutu (augment=True ise kullanılır).
        split: Degradation split adı (light/medium/heavy/extreme). None ise
            root_dir/clean + root_dir/degraded kullanılır; belirtilirse
            root_dir/{split}/clean + root_dir/{split}/degraded kullanılır.
    """

    def __init__(
        self,
        root_dir,
        use_minio=False,
        bucket_name="datasets",
        minio_prefix="datasets/training-512/v1",
        augment=False,
        crop_size=256,
        split=None,
    ):
        self.root_dir = Path(root_dir)
        self.use_minio = use_minio
        self.bucket_name = bucket_name
        self.minio_prefix = minio_prefix
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

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, index):
        if not self.use_minio:
            clean_data = np.load(self.clean_files[index])
            degraded_data = np.load(self.degraded_files[index])
        else:
            clean_data = load_npy_from_minio(self.bucket_name, self.clean_files[index])
            degraded_data = load_npy_from_minio(
                self.bucket_name, self.degraded_files[index]
            )

        clean = torch.from_numpy(clean_data)
        degraded = torch.from_numpy(degraded_data)

        clean = clean.unsqueeze(0)
        degraded = degraded.unsqueeze(0)

        if self.augment:
            clean, degraded = self._augment(clean, degraded)

        return degraded, clean

    def _augment(self, degraded, clean):
        """Deterministic augmentation: random flip + 90° rotation + random crop.

        Her çağrıda yeni bir Generator oluşturulur — epoch başına farklı ama
        aynı epoch içinde aynı index için aynı sonucu verir (DataLoader shuffle
        ile birlikte her epoch'ta farklı augmentasyon görülür).
        """
        gen = torch.Generator()

        # Random horizontal flip
        if torch.rand(1, generator=gen).item() < 0.5:
            degraded = torch.flip(degraded, dims=[-1])
            clean = torch.flip(clean, dims=[-1])

        # Random vertical flip
        if torch.rand(1, generator=gen).item() < 0.5:
            degraded = torch.flip(degraded, dims=[-2])
            clean = torch.flip(clean, dims=[-2])

        # Random 90° rotation (k ∈ {0, 1, 2, 3})
        k = int(torch.randint(0, 4, (1,), generator=gen).item())
        if k > 0:
            degraded = torch.rot90(degraded, k=k, dims=[-2, -1])
            clean = torch.rot90(clean, k=k, dims=[-2, -1])

        # Random crop (crop_size × crop_size) — padding ile sınır dışı korunur
        # Handle both 3D (C, H, W) and 4D (N, C, H, W) tensors
        if degraded.dim() == 3:
            _, h, w = degraded.shape
        else:
            _, _, h, w = degraded.shape
        crop = self.crop_size
        if h >= crop and w >= crop:
            top = int(torch.randint(0, h - crop + 1, (1,), generator=gen).item())
            left = int(torch.randint(0, w - crop + 1, (1,), generator=gen).item())
            degraded = degraded[..., top : top + crop, left : left + crop]
            clean = clean[..., top : top + crop, left : left + crop]

        return degraded, clean
