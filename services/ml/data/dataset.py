import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from services.ml.data.minio_loader import load_npy_from_minio, list_files_in_minio

class BlackHoleDataset(Dataset):
    def __init__(self, root_dir, use_minio=False, bucket_name="karadelikler", minio_prefix="datasets/training-512/v1"):
        self.root_dir = Path(root_dir)
        self.use_minio = use_minio
        self.bucket_name = bucket_name
        self.minio_prefix = minio_prefix

        if not self.use_minio:
            self.clean_files = sorted(
                (self.root_dir / "clean").glob("*.npy")
            )
            self.degraded_files = sorted(
                (self.root_dir / "degraded").glob("*.npy")
            )
        else:
            clean_path = f"{self.minio_prefix}/clean/"
            degraded_path = f"{self.minio_prefix}/degraded/"

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
            degraded_data = load_npy_from_minio(self.bucket_name, self.degraded_files[index])

        clean = torch.from_numpy(clean_data)
        degraded = torch.from_numpy(degraded_data)

        clean = clean.unsqueeze(0)
        degraded = degraded.unsqueeze(0)

        return degraded, clean