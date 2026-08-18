import time
from pathlib import Path

import hydra
import mlflow
import mlflow.pytorch
import torch
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast

from services.ml.evaluation.benchmark import (
    ValidationSummary,
    evaluate_validation_loader,
    save_sample_outputs,
    save_validation_summary,
    update_best_model,
)
from services.ml.data.dataloader import create_train_val_loaders
from services.ml.models.unet import UNet
from services.ml.losses.loss import get_loss
from services.ml.checkpoints.checkpoint import load_checkpoint, save_checkpoint


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> Path:
    """Train U-Net baseline on black hole image pairs.

    All hyperparameters come from Hydra config (services/ml/conf/*.yaml).
    Override from CLI: `python -m services.ml.training.train training.epochs=2`

    Every run is logged to MLflow: params (full config), per-epoch metrics,
    and artifacts (best_model.pt, sample PNGs, validation_results.jsonl).
    """
    # Resolve device
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    # Output dir: Hydra already created outputs/<date>/<time>/; use cfg.paths.output_dir
    # relative to that working dir so checkpoints land alongside the config snapshot.
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    torch.manual_seed(cfg.seed)

    # Data
    train_loader, val_loader = create_train_val_loaders(
        root_dir=cfg.paths.root_dir,
        batch_size=cfg.training.batch_size,
        val_ratio=cfg.training.val_ratio,
        use_minio=cfg.data.use_minio,
        bucket_name=cfg.data.bucket_name,
        minio_prefix=cfg.data.minio_prefix,
        augment=cfg.data.augment,
        crop_size=cfg.data.crop_size,
        split=cfg.data.get("split", None),
        max_samples=cfg.data.get("max_samples", None),
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=cfg.data.drop_last,
    )

    # Model
    model = UNet().to(device)

    # Loss — factory pattern: mse | l1 | smooth_l1 (Faz 2'de perceptual/gan/physics)
    criterion = get_loss(cfg.loss.name)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        betas=tuple(cfg.training.optimizer.betas),
        weight_decay=cfg.training.optimizer.weight_decay,
    )

    scheduler_name = str(cfg.training.scheduler.name)
    if scheduler_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(cfg.training.scheduler.factor),
            patience=int(cfg.training.scheduler.patience),
            min_lr=float(cfg.training.scheduler.min_lr),
        )
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.training.epochs),
            eta_min=float(cfg.training.scheduler.min_lr),
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    # AMP (Mixed Precision) — cfg.training.amp=true ise FP16/BF16 ile eğitim
    # L40S BF16 destekliyor (daha kararlı, FP16'dan overflow riski düşük)
    use_amp = bool(cfg.training.amp) and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16 if cfg.training.amp_dtype == "bfloat16" else torch.float16
    )
    # GradScaler sadece FP16 için gerekli (BF16'da scaler no-op)
    scaler = GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

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

    # MLflow setup — tracking_uri ve experiment_name config'den
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # run_name None ise MLflow otomatik üretir; aksi halde config'den okunur
    run_name = cfg.mlflow.get("run_name") or None

    with mlflow.start_run(run_name=run_name):
        # Tüm config'i params olarak logla (nested dict'ler düzleştirilir)
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        model_input_example = None
        training_started_at = time.monotonic()

        for epoch in range(start_epoch, cfg.training.epochs):
            epoch_started_at = time.monotonic()
            model.train()
            running_train_loss = 0.0

            # Gradient accumulation: effective batch = batch_size * grad_accum_steps
            # grad_accum_steps=1 → eski davranış (her batch'te step)
            grad_accum_steps = max(1, int(cfg.training.grad_accum_steps))

            for batch_idx, (degraded, clean) in enumerate(train_loader):
                degraded = degraded.to(device, dtype=torch.float32)
                clean = clean.to(device, dtype=torch.float32)
                if model_input_example is None:
                    model_input_example = degraded[:1].detach().cpu().numpy()

                # AMP forward pass — autocast context manager ile
                # torch 2.2.x: torch.cuda.amp.autocast(enabled=..., dtype=...)
                with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                    prediction = model(degraded)
                    loss_result = criterion(prediction, clean)
                    loss = (
                        loss_result["total"]
                        if isinstance(loss_result, dict)
                        else loss_result
                    )
                    # Accumulation: loss'u grad_accum_steps'e böl (gradient'lerin
                    # toplamı effective batch loss'una eşit olsun)
                    loss = loss / grad_accum_steps

                # AMP backward — gradyanları biriktir, henüz step atma
                # BF16'da scaler no-op; FP16'da loss'u scale eder
                if use_amp and amp_dtype == torch.float16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Her grad_accum_steps batch'te bir optimizer step
                is_accum_step = (batch_idx + 1) % grad_accum_steps == 0
                if is_accum_step:
                    if use_amp and amp_dtype == torch.float16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                # Loss'u orijinal (ölçeklenmemiş) değer olarak logla
                running_train_loss += float(loss.item()) * grad_accum_steps

            if len(train_loader) % grad_accum_steps != 0:
                if use_amp and amp_dtype == torch.float16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            train_loss = running_train_loss / max(1, len(train_loader))

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

            # Per-epoch metrics → MLflow
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
                # Best model artifact'larını MLflow'a logla
                mlflow.log_artifact(str(best_checkpoint_path))

            # Her epoch sonunda sample PNG ve jsonl'i logla
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
