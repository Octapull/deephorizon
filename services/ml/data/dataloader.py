import torch
from torch.utils.data import DataLoader, random_split

from .dataset import BlackHoleDataset


def _create_loader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_dataloader(
    root_dir,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
):
    dataset = BlackHoleDataset(root_dir)
    return _create_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_train_val_loaders(
    root_dir,
    batch_size=16,
    val_ratio=0.2,
    seed=42,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    dataset = BlackHoleDataset(root_dir)
    dataset_size = len(dataset)

    if dataset_size < 2:
        raise ValueError("At least 2 samples are required to create train/val splits")

    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size

    if train_size == 0 or val_size == 0:
        raise ValueError("The selected val_ratio results in an empty split")

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = _create_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_loader = _create_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader
