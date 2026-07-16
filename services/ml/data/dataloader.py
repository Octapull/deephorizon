from torch.utils.data import DataLoader
from .dataset import BlackHoleDataset


def create_dataloader(root_dir, batch_size=16):

    dataset = BlackHoleDataset(root_dir)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    return loader
