from pathlib import Path
import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    psnr: float,
    ssim: float,
    checkpoint_path,
    scheduler=None,
):
    """Save model + optimizer + metrics to a checkpoint file.

    Args:
        model: PyTorch model (state_dict saved).
        optimizer: PyTorch optimizer (state_dict saved).
        epoch: Current epoch number (1-indexed).
        train_loss: Training loss for this epoch.
        val_loss: Validation loss for this epoch.
        psnr: Validation PSNR for this epoch.
        ssim: Validation SSIM for this epoch.
        checkpoint_path: Where to write the .pt file.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "psnr": psnr,
            "ssim": ssim,
        }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(payload, checkpoint_path)


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    map_location="cpu",
    scheduler=None,
):
    """Load model + optimizer + metrics from a checkpoint file.

    Returns the full checkpoint dict. Missing keys (e.g. older checkpoints
    saved before psnr/ssim were added) default to 0.0 for backward compat.

    Args:
        checkpoint_path: Path to the .pt file.
        model: PyTorch model (state_dict loaded into).
        optimizer: PyTorch optimizer (state_dict loaded into, optional).
        map_location: Device to map tensors to (default "cpu").

    Returns:
        dict with keys: epoch, model_state_dict, optimizer_state_dict,
        train_loss, val_loss, psnr, ssim.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Backward compat: eski checkpoint'lerde psnr/ssim yoksa 0.0 döndür
    checkpoint.setdefault("psnr", 0.0)
    checkpoint.setdefault("ssim", 0.0)

    return checkpoint
