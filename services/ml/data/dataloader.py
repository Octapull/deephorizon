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
    use_minio=False,
    bucket_name="datasets",
    minio_prefix="training-512/v1",
    augment=False,
    crop_size=256,
    split=None,
    max_samples=None,
):
    dataset = BlackHoleDataset(
        root_dir,
        use_minio=use_minio,
        bucket_name=bucket_name,
        minio_prefix=minio_prefix,
        augment=augment,
        crop_size=crop_size,
        split=split,
        max_samples=max_samples,
    )
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
    use_minio=False,
    bucket_name="datasets",
    minio_prefix="training-512/v1",
    augment=False,
    crop_size=256,
    split=None,
    max_samples=None,
):
    """Train + validation DataLoader oluşturur.

    Args:
        root_dir: Local kök dizin (use_minio=False ise).
        use_minio: True ise MinIO/S3'ten okur.
        bucket_name: MinIO bucket adı.
        minio_prefix: MinIO prefix.
        augment: True ise train set'e augmentation uygulanır (val set'e uygulanmaz).
        crop_size: Random crop boyutu (augment=True ise).
        split: Degradation split adı (light/medium/heavy/extreme).
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    # Train dataset — augmentation açık
    train_full = BlackHoleDataset(
        root_dir,
        use_minio=use_minio,
        bucket_name=bucket_name,
        minio_prefix=minio_prefix,
        augment=augment,
        crop_size=crop_size,
        split=split,
        max_samples=max_samples,
    )
    dataset_size = len(train_full)

    if dataset_size < 2:
        raise ValueError("At least 2 samples are required to create train/val splits")

    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size

    if train_size == 0 or val_size == 0:
        raise ValueError("The selected val_ratio results in an empty split")

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_full, [train_size, val_size], generator=generator
    )

    # Validation dataset — augmentation KAPALI (deterministic ölçüm)
    val_full = BlackHoleDataset(
        root_dir,
        use_minio=use_minio,
        bucket_name=bucket_name,
        minio_prefix=minio_prefix,
        augment=False,
        crop_size=crop_size,
        split=split,
        max_samples=max_samples,
    )
    # Aynı index'leri kullanmak için val_subset'in index'lerini val_full'e uygula
    val_indices = val_subset.indices
    val_dataset = torch.utils.data.Subset(val_full, val_indices)

    train_loader = _create_loader(
        train_subset,
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
