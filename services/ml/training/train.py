from pathlib import Path

import hydra
import mlflow
import mlflow.pytorch
import torch
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast

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
from services.ml.checkpoints.checkpoint import save_checkpoint


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

    # MLflow setup — tracking_uri ve experiment_name config'den
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # run_name None ise MLflow otomatik üretir; aksi halde config'den okunur
    run_name = cfg.mlflow.get("run_name") or None

    with mlflow.start_run(run_name=run_name):
        # Tüm config'i params olarak logla (nested dict'ler düzleştirilir)
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        for epoch in range(cfg.training.epochs):
            model.train()
            running_train_loss = 0.0

            # Gradient accumulation: effective batch = batch_size * grad_accum_steps
            # grad_accum_steps=1 → eski davranış (her batch'te step)
            grad_accum_steps = max(1, int(cfg.training.grad_accum_steps))

            for batch_idx, (degraded, clean) in enumerate(train_loader):
                degraded = degraded.to(device, dtype=torch.float32)
                clean = clean.to(device, dtype=torch.float32)

                # AMP forward pass — autocast context manager ile
                # torch 2.2.x: torch.cuda.amp.autocast(enabled=..., dtype=...)
                with autocast(enabled=use_amp, dtype=amp_dtype):
                    prediction = model(degraded)
                    loss = criterion(prediction, clean)
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

            # Per-epoch metrics → MLflow
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": validation_summary.val_loss,
                    "psnr": validation_summary.psnr,
                    "ssim": validation_summary.ssim,
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
            )

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
                if cfg.mlflow.get("log_model", True):
                    mlflow.pytorch.log_model(model, "best_model")

            # Her epoch sonunda sample PNG ve jsonl'i logla
            if sample_dir is not None:
                mlflow.log_artifact(str(sample_dir), artifact_path="samples")
            mlflow.log_artifact(str(jsonl_path), artifact_path="metrics")

    return best_checkpoint_path


if __name__ == "__main__":
    train()
