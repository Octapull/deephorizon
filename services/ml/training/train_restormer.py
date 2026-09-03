"""Restormer training loop for image restoration.

Trains a Restormer model (Zamir et al., 2022) using a patch-based
approach: 512×512 images are cropped to ``patch_size×patch_size``
patches during training to fit in GPU memory. At inference, the
full image is processed (or tiled with overlap if needed).

Key differences from ``train.py`` (U-Net baseline):

1. **Patch-based training**: Random crops of ``patch_size`` instead of
   full 512×512 images. This is the approach recommended in the
   Restormer paper for high-resolution image restoration.
2. **Full loss suite**: L1 (pixel) + perceptual (VGG) + physics
   (flux + ring + asymmetry) via the ``combined`` loss factory.
3. **Lower learning rate**: Restormer is sensitive to LR; 1e-4 is
   the paper default for transformer-based restoration models.
4. **Cosine scheduler**: Restormer paper uses cosine annealing
   rather than ReduceLROnPlateau.

Reference:
    Zamir, S. W., Arora, A., Khan, S., Hayat, M., Khan, F. S., Yang, M.-H.,
    Shao, L. (2022). "Restormer: Efficient Transformer for High-Resolution
    Image Restoration." CVPR.
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
import mlflow
import mlflow.pytorch
import torch
from omegaconf import DictConfig, OmegaConf

# PyTorch 2.2.x uyumluluğu: torch.amp.GradScaler 2.3+ ile gelir.
try:
    from torch.amp import GradScaler, autocast

    _AMP_HAS_DTYPE = True
except ImportError:  # pragma: no cover - eski torch sürümleri için
    from torch.cuda.amp import GradScaler, autocast

    _AMP_HAS_DTYPE = False

from services.ml.checkpoints.checkpoint import load_checkpoint, save_checkpoint
from services.ml.data.dataloader import create_train_val_loaders
from services.ml.evaluation.benchmark import (
    ValidationSummary,
    evaluate_validation_loader,
    save_sample_outputs,
    save_validation_summary,
    update_best_model,
)
from services.ml.losses.loss import get_loss
from services.ml.models.restormer import build_restormer


def _format_duration(seconds: float) -> str:
    """Format seconds as ``Hh Mm Ss`` for human-readable logging."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _random_crop(
    degraded: torch.Tensor,
    clean: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random spatial crop for patch-based training.

    Restormer paper trains on ``patch_size×patch_size`` crops rather
    than full 512×512 images. This reduces GPU memory by ~16× and
    acts as a form of data augmentation (the model sees different
    spatial regions each epoch).

    Args:
        degraded: Degraded input batch, shape ``(B, C, H, W)``.
        clean: Clean target batch, shape ``(B, C, H, W)``.
        patch_size: Side length of the square crop.

    Returns:
        Tuple of ``(degraded_crop, clean_crop)``, both with shape
        ``(B, C, patch_size, patch_size)``.
    """
    _, _, h, w = degraded.shape
    if h < patch_size or w < patch_size:
        # Görüntü patch'ten küçükse olduğu gibi döndür (edge case)
        return degraded, clean

    # Rastgele sol-üst köşe seç
    top = torch.randint(0, h - patch_size + 1, (1,)).item()
    left = torch.randint(0, w - patch_size + 1, (1,)).item()

    degraded_crop = degraded[:, :, top : top + patch_size, left : left + patch_size]
    clean_crop = clean[:, :, top : top + patch_size, left : left + patch_size]
    return degraded_crop, clean_crop


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> Path:
    """Train Restormer on black hole image pairs.

    All hyperparameters come from Hydra config. Override from CLI:

        python -m services.ml.training.train_restormer \\
            model=restormer training.epochs=300

    Every run is logged to MLflow: params (full config), per-epoch
    metrics, and artifacts (best_model.pt, sample PNGs, validation
    results).
    """
    # ─── Device ───
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    # ─── Output dir ───
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Reproducibility ───
    torch.manual_seed(cfg.seed)

    # ─── Data ───
    # crop_size=None → tam 512×512 yükle, patch-based crop'u biz yapacağız
    train_loader, val_loader = create_train_val_loaders(
        root_dir=cfg.paths.root_dir,
        batch_size=cfg.training.batch_size,
        val_ratio=cfg.training.val_ratio,
        use_minio=cfg.data.use_minio,
        bucket_name=cfg.data.bucket_name,
        minio_prefix=cfg.data.minio_prefix,
        augment=cfg.data.augment,
        crop_size=512,  # tam 512×512 yükle, patch-based crop'u train loop'ta yap
        split=cfg.data.get("split", None),
        max_samples=cfg.data.get("max_samples", None),
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=cfg.data.drop_last,
    )

    # ─── Model ───
    # Restormer config'den: dim, num_blocks, num_heads, expansion_factor
    model_cfg = cfg.model
    model = build_restormer(
        in_channels=int(model_cfg.in_channels),
        out_channels=int(model_cfg.out_channels),
        dim=int(model_cfg.dim),
        num_blocks=list(model_cfg.num_blocks),
        num_heads=list(model_cfg.num_heads),
        expansion_factor=float(model_cfg.expansion_factor),
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Restormer parameters: {total_params:,}")

    # ─── Loss ───
    # Full loss suite: L1 + perceptual + physics (combined factory)
    criterion = get_loss(cfg.loss.name)

    # ─── Optimizer ───
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        betas=tuple(cfg.training.optimizer.betas),
        weight_decay=cfg.training.optimizer.weight_decay,
    )

    # ─── Scheduler ───
    # Restormer paper: cosine annealing
    scheduler_name = str(cfg.training.scheduler.name)
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.training.epochs),
            eta_min=float(cfg.training.scheduler.min_lr),
        )
    elif scheduler_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(cfg.training.scheduler.factor),
            patience=int(cfg.training.scheduler.patience),
            min_lr=float(cfg.training.scheduler.min_lr),
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    # ─── AMP ───
    use_amp = bool(cfg.training.amp) and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16 if cfg.training.amp_dtype == "bfloat16" else torch.float16
    )
    scaler = GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    # ─── Patch size ───
    patch_size = int(getattr(model_cfg, "patch_size", 128))

    # ─── Checkpoint resume ───
    best_val_loss = float("inf")
    best_checkpoint_path = output_dir / "best_model.pt"
    start_epoch = 0

    resume_from = cfg.training.get("resume_from")
    if resume_from:
        checkpoint = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = int(checkpoint["epoch"])
        best_val_loss = float(checkpoint.get("val_loss", best_val_loss))
        print(f"Resuming from {resume_from} at epoch {start_epoch + 1}")

    early_stop_best = best_val_loss
    epochs_without_improvement = 0

    # ─── MLflow ───
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    run_name = cfg.mlflow.get("run_name") or None

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        mlflow.log_param("total_params", total_params)
        model_input_example = None
        training_started_at = time.monotonic()

        for epoch in range(start_epoch, cfg.training.epochs):
            epoch_started_at = time.monotonic()
            model.train()
            running_train_loss = 0.0

            grad_accum_steps = max(1, int(cfg.training.grad_accum_steps))

            for batch_idx, (degraded, clean) in enumerate(train_loader):
                degraded = degraded.to(device, dtype=torch.float32)
                clean = clean.to(device, dtype=torch.float32)

                # Patch-based training: rastgele crop
                degraded, clean = _random_crop(degraded, clean, patch_size)

                if model_input_example is None:
                    model_input_example = degraded[:1].detach().cpu().numpy()

                with autocast(
                    device_type=device.type, enabled=use_amp, dtype=amp_dtype
                ):
                    prediction = model(degraded)
                    loss_result = criterion(prediction, clean)
                    loss = (
                        loss_result["total"]
                        if isinstance(loss_result, dict)
                        else loss_result
                    )
                    loss = loss / grad_accum_steps

                if use_amp and amp_dtype == torch.float16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                is_accum_step = (batch_idx + 1) % grad_accum_steps == 0
                if is_accum_step:
                    if use_amp and amp_dtype == torch.float16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                running_train_loss += float(loss.item()) * grad_accum_steps

            if len(train_loader) % grad_accum_steps != 0:
                if use_amp and amp_dtype == torch.float16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            train_loss = running_train_loss / max(1, len(train_loader))

            # ─── Validation ───
            validation_summary, sample_batch = evaluate_validation_loader(
                model=model,
                val_loader=val_loader,
                device=device,
                criterion=criterion,
            )
            validation_summary = ValidationSummary(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=validation_summary.val_loss,
                psnr=validation_summary.psnr,
                ssim=validation_summary.ssim,
            )

            if scheduler is not None:
                if scheduler_name == "reduce_on_plateau":
                    scheduler.step(validation_summary.val_loss)
                else:
                    scheduler.step()

            learning_rate = float(optimizer.param_groups[0]["lr"])

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": validation_summary.val_loss,
                    "psnr": validation_summary.psnr,
                    "ssim": validation_summary.ssim,
                    "learning_rate": learning_rate,
                },
                step=epoch + 1,
            )

            sample_dir = None
            if sample_batch is not None:
                sample_dir = save_sample_outputs(
                    degraded=sample_batch[0],
                    prediction=sample_batch[1],
                    clean=sample_batch[2],
                    output_dir=output_dir,
                    epoch=epoch + 1,
                )

            jsonl_path = save_validation_summary(validation_summary, output_dir)

            print(
                f"Epoch [{epoch + 1}/{cfg.training.epochs}] "
                f"Train Loss: {train_loss:.6f} "
                f"Val Loss: {validation_summary.val_loss:.6f} "
                f"PSNR: {validation_summary.psnr:.4f} "
                f"SSIM: {validation_summary.ssim:.4f}"
            )

            # ─── Checkpoint ───
            checkpoint_path = output_dir / f"epoch_{epoch + 1}.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=validation_summary.val_loss,
                psnr=validation_summary.psnr,
                ssim=validation_summary.ssim,
                checkpoint_path=checkpoint_path,
                scheduler=scheduler,
            )

            keep_last_n = max(1, int(cfg.training.keep_last_n))
            epoch_checkpoints = sorted(
                output_dir.glob("epoch_*.pt"),
                key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
            )
            for stale_checkpoint in epoch_checkpoints[:-keep_last_n]:
                stale_checkpoint.unlink()

            best_val_loss, is_best, best_checkpoint_path = update_best_model(
                summary=validation_summary,
                model=model,
                optimizer=optimizer,
                output_dir=output_dir,
                best_val_loss=best_val_loss,
            )
            if is_best:
                print(f"Best model updated: {best_checkpoint_path}")
                mlflow.log_artifact(str(best_checkpoint_path))

            if sample_dir is not None:
                mlflow.log_artifact(str(sample_dir), artifact_path="samples")
            mlflow.log_artifact(str(jsonl_path), artifact_path="metrics")

            epoch_duration = time.monotonic() - epoch_started_at
            completed_in_session = epoch - start_epoch + 1
            average_epoch_duration = (
                time.monotonic() - training_started_at
            ) / completed_in_session
            remaining_epochs = int(cfg.training.epochs) - (epoch + 1)
            estimated_remaining = average_epoch_duration * remaining_epochs
            mlflow.log_metrics(
                {
                    "epoch_duration_seconds": epoch_duration,
                    "estimated_remaining_seconds": estimated_remaining,
                },
                step=epoch + 1,
            )
            print(
                f"Timing: epoch={_format_duration(epoch_duration)} "
                f"average={_format_duration(average_epoch_duration)} "
                f"eta_max={_format_duration(estimated_remaining)}"
            )

            # ─── Early stopping ───
            early_stopping = cfg.training.early_stopping
            min_delta = float(early_stopping.min_delta)
            if validation_summary.val_loss < early_stop_best - min_delta:
                early_stop_best = validation_summary.val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (
                bool(early_stopping.enabled)
                and epoch + 1 >= int(early_stopping.min_epochs)
                and epochs_without_improvement >= int(early_stopping.patience)
            ):
                print(
                    "Early stopping: validation loss did not improve for "
                    f"{epochs_without_improvement} epochs."
                )
                break

        # ─── Final model log ───
        if (
            cfg.mlflow.get("log_model", True)
            and best_checkpoint_path.exists()
            and model_input_example is not None
        ):
            load_checkpoint(best_checkpoint_path, model=model, map_location=device)
            model.to("cpu")
            model.eval()
            mlflow.pytorch.log_model(
                model,
                name="best_model",
                input_example=model_input_example,
                serialization_format="pickle",
            )

    return best_checkpoint_path


if __name__ == "__main__":
    train()
