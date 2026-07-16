import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class BlackHoleDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)

        self.clean_files = sorted(
            (self.root_dir / "clean").glob("*.npy")
        )

        self.degraded_files = sorted(
            (self.root_dir / "degraded").glob("*.npy")
        )


    def __len__(self):
        return len(self.clean_files)


    def __getitem__(self, index):

        clean = np.load(self.clean_files[index])
        degraded = np.load(self.degraded_files[index])

        clean = torch.from_numpy(clean)
        degraded = torch.from_numpy(degraded)

        clean = clean.unsqueeze(0)
        degraded = degraded.unsqueeze(0)

        return degraded, clean