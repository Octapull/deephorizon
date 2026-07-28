"""GAN training loop for Pix2Pix image-to-image translation.

Trains a generator (Pix2PixGenerator) and discriminator (PatchDiscriminator)
in alternating fashion using the CombinedLoss objective. Supports:

- Mixed precision (AMP) with BF16/FP16
- Gradient accumulation
- MLflow experiment tracking
- Hydra configuration
- Checkpoint save/load for both G and D
- Best-model tracking by validation loss

The training follows the standard Pix2Pix recipe:
    1. Forward pass: fake = G(condition)
    2. Discriminator step: minimize D loss on (real, fake)
    3. Generator step: minimize G loss (pixel + perceptual + adv + physics)
    4. Log metrics, save checkpoint, repeat

Reference:
    Isola, P. et al. (2017). "Image-to-Image Translation with
    Conditional Adversarial Networks." CVPR.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
import mlflow.pytorch
import torch
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast

from services.ml.checkpoints.checkpoint import save_checkpoint
from services.ml.data.dataloader import create_train_val_loaders
from services.ml.evaluation.benchmark import (
    ValidationSummary,
    evaluate_validation_loader,
    save_sample_outputs,
    save_validation_summary,
)
from services.ml.losses.combined import CombinedLoss
from services.ml.losses.gan import DiscriminatorAdversarialLoss
from services.ml.models.pix2pix import PatchDiscriminator, Pix2PixGenerator


def _build_generator(cfg: DictConfig, device: torch.device) -> Pix2PixGenerator:
    """Build the Pix2Pix generator from config."""
    model_cfg = cfg.model
    generator = Pix2PixGenerator(
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        dropout=float(model_cfg.get("dropout", 0.5)),
        use_tanh=bool(model_cfg.get("use_tanh", True)),
    )
    return generator.to(device)


def _build_discriminator(cfg: DictConfig, device: torch.device) -> PatchDiscriminator:
    """Build the PatchGAN discriminator from config."""
    disc_cfg = cfg.model.get("discriminator", {})
    discriminator = PatchDiscriminator(
        in_channels=int(disc_cfg.get("in_channels", 2)),
        base_channels=int(disc_cfg.get("base_channels", 64)),
        max_channels=int(disc_cfg.get("max_channels", 512)),
        n_layers=int(disc_cfg.get("n_layers", 3)),
        use_sigmoid=bool(disc_cfg.get("use_sigmoid", True)),
    )
    return discriminator.to(device)


def _build_losses(cfg: DictConfig) -> tuple[CombinedLoss, DiscriminatorAdversarialLoss]:
    """Build generator (combined) and discriminator losses from config."""
    loss_cfg = cfg.loss
    weights = loss_cfg.weights

    generator_loss = CombinedLoss(
        pixel_weight=float(weights.get("pixel", 100.0)),
        perceptual_weight=float(weights.get("perceptual", 0.0)),
        adversarial_weight=float(weights.get("adversarial", 1.0)),
        physics_weight=float(weights.get("physics", 0.0)),
        pixel_loss=str(loss_cfg.get("pixel_loss", "l1")),
        gan_mode=str(loss_cfg.get("gan_mode", "lsgan")),
        perceptual_layer=str(loss_cfg.get("perceptual_layer", "relu2_2")),
    )

    discriminator_loss = DiscriminatorAdversarialLoss(
        mode=str(loss_cfg.get("gan_mode", "lsgan")),
    )

    return generator_loss, discriminator_loss


def _build_optimizers(
    cfg: DictConfig,
    generator: Pix2PixGenerator,
    discriminator: PatchDiscriminator,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """Build Adam optimizers for G and D.

    Following Pix2Pix, both networks use the same learning rate and
    betas. The discriminator learning rate is typically halved to slow
    down D relative to G.
    """
    opt_cfg = cfg.training.optimizer
    lr = float(cfg.training.learning_rate)
    betas = tuple(opt_cfg.betas)
    weight_decay = float(opt_cfg.weight_decay)

    d_lr = lr * float(cfg.training.get("discriminator_lr_factor", 0.5))

    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=d_lr,
        betas=betas,
        weight_decay=weight_decay,
    )
    return g_optimizer, d_optimizer


def _train_discriminator_step(
    discriminator: PatchDiscriminator,
    d_optimizer: torch.optim.Optimizer,
    d_loss_fn: DiscriminatorAdversarialLoss,
    condition: torch.Tensor,
    real_target: torch.Tensor,
    fake_target: torch.Tensor,
    use_amp: bool,
    amp_dtype: torch.dtype,
    scaler: GradScaler | None,
) -> float:
    """Single discriminator training step.

    Args:
        discriminator: PatchGAN discriminator.
        d_optimizer: Discriminator optimizer.
        d_loss_fn: Discriminator adversarial loss.
        condition: Conditional input (degraded image).
        real_target: Real (clean) target image.
        fake_target: Generated (fake) target image (detached).
        use_amp: Whether to use mixed precision.
        amp_dtype: AMP dtype (bfloat16 or float16).
        scaler: GradScaler for FP16 (None for BF16).

    Returns:
        Discriminator loss value (float).
    """
    discriminator.train()
    d_optimizer.zero_grad()

    with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        disc_real = discriminator(condition, real_target)
        disc_fake = discriminator(condition, fake_target)
        d_loss = d_loss_fn(disc_real, disc_fake)

    if use_amp and amp_dtype == torch.float16 and scaler is not None:
        scaler.scale(d_loss).backward()
        scaler.step(d_optimizer)
        scaler.update()
    else:
        d_loss.backward()
        d_optimizer.step()

    return float(d_loss.item())


def _train_generator_step(
    generator: Pix2PixGenerator,
    discriminator: PatchDiscriminator,
    g_optimizer: torch.optim.Optimizer,
    g_loss_fn: CombinedLoss,
    condition: torch.Tensor,
    real_target: torch.Tensor,
    use_amp: bool,
    amp_dtype: torch.dtype,
    scaler: GradScaler | None,
) -> dict[str, float]:
    """Single generator training step.

    Args:
        generator: Pix2Pix generator.
        discriminator: PatchGAN discriminator (frozen during G step).
        g_optimizer: Generator optimizer.
        g_loss_fn: Combined loss (pixel + perceptual + adv + physics).
        condition: Conditional input (degraded image).
        real_target: Real (clean) target image.
        use_amp: Whether to use mixed precision.
        amp_dtype: AMP dtype.
        scaler: GradScaler for FP16.

    Returns:
        Dict with loss components: total, pixel, perceptual, adversarial, physics.
    """
    generator.train()
    g_optimizer.zero_grad()

    with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        fake_target = generator(condition)
        # Discriminator output for the fake (no grad through D)
        disc_fake = discriminator(condition, fake_target)
        components = g_loss_fn(fake_target, real_target, disc_output=disc_fake)
        g_loss = components["total"]

    if use_amp and amp_dtype == torch.float16 and scaler is not None:
        scaler.scale(g_loss).backward()
        scaler.step(g_optimizer)
        scaler.update()
    else:
        g_loss.backward()
        g_optimizer.step()

    return {k: float(v.item()) for k, v in components.items()}


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train_gan(cfg: DictConfig) -> Path:
    """Train Pix2Pix GAN on black hole image pairs.

    All hyperparameters come from Hydra config (services/ml/conf/*.yaml).
    Override from CLI:
        python -m services.ml.training.gan_train training.epochs=2

    Every run is logged to MLflow: params (full config), per-epoch metrics
    (G loss, D loss, PSNR, SSIM), and artifacts (best_model.pt, sample PNGs).
    """
    # Resolve device
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    # Output dir
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
    )

    # Models
    generator = _build_generator(cfg, device)
    discriminator = _build_discriminator(cfg, device)

    # Losses
    generator_loss, discriminator_loss = _build_losses(cfg)

    # Optimizers
    g_optimizer, d_optimizer = _build_optimizers(cfg, generator, discriminator)

    # AMP
    use_amp = bool(cfg.training.amp) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.training.amp_dtype == "bfloat16" else torch.float16
    scaler = GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    # Gradient accumulation
    grad_accum_steps = max(1, int(cfg.training.grad_accum_steps))

    best_val_loss = float("inf")
    best_checkpoint_path = output_dir / "best_model.pt"

    # MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    run_name = cfg.mlflow.get("run_name") or None

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        for epoch in range(cfg.training.epochs):
            generator.train()
            discriminator.train()

            running_g_total = 0.0
            running_g_pixel = 0.0
            running_g_perceptual = 0.0
            running_g_adversarial = 0.0
            running_g_physics = 0.0
            running_d_loss = 0.0
            batch_count = 0

            for batch_idx, (degraded, clean) in enumerate(train_loader):
                degraded = degraded.to(device, dtype=torch.float32)
                clean = clean.to(device, dtype=torch.float32)

                # ---- Generator step ----
                # Generate fake images (no grad yet — D step needs detached fakes)
                generator.eval()  # disable dropout for D input
                with torch.no_grad():
                    fake_for_d = generator(degraded)
                generator.train()

                # D step on (real, fake)
                d_loss = _train_discriminator_step(
                    discriminator=discriminator,
                    d_optimizer=d_optimizer,
                    d_loss_fn=discriminator_loss,
                    condition=degraded,
                    real_target=clean,
                    fake_target=fake_for_d.detach(),
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    scaler=scaler,
                )

                # G step (re-enables grad through generator)
                g_components = _train_generator_step(
                    generator=generator,
                    discriminator=discriminator,
                    g_optimizer=g_optimizer,
                    g_loss_fn=generator_loss,
                    condition=degraded,
                    real_target=clean,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    scaler=scaler,
                )

                # Accumulate metrics
                running_g_total += g_components["total"]
                running_g_pixel += g_components.get("pixel", 0.0)
                running_g_perceptual += g_components.get("perceptual", 0.0)
                running_g_adversarial += g_components.get("adversarial", 0.0)
                running_g_physics += g_components.get("physics", 0.0)
                running_d_loss += d_loss
                batch_count += 1

            # Average metrics
            avg_g_total = running_g_total / max(1, batch_count)
            avg_g_pixel = running_g_pixel / max(1, batch_count)
            avg_g_perceptual = running_g_perceptual / max(1, batch_count)
            avg_g_adversarial = running_g_adversarial / max(1, batch_count)
            avg_g_physics = running_g_physics / max(1, batch_count)
            avg_d_loss = running_d_loss / max(1, batch_count)

            # ---- Validation ----
            # Use only the generator for validation (pixel loss only)
            generator.eval()
            val_criterion = generator_loss._pixel  # type: ignore[attr-defined]
            validation_summary, sample_batch = evaluate_validation_loader(
                model=generator,
                val_loader=val_loader,
                device=device,
                criterion=val_criterion,
            )
            validation_summary = ValidationSummary(
                epoch=epoch + 1,
                train_loss=avg_g_total,
                val_loss=validation_summary.val_loss,
                psnr=validation_summary.psnr,
                ssim=validation_summary.ssim,
            )

            # MLflow logging
            mlflow.log_metrics(
                {
                    "g_total": avg_g_total,
                    "g_pixel": avg_g_pixel,
                    "g_perceptual": avg_g_perceptual,
                    "g_adversarial": avg_g_adversarial,
                    "g_physics": avg_g_physics,
                    "d_loss": avg_d_loss,
                    "val_loss": validation_summary.val_loss,
                    "psnr": validation_summary.psnr,
                    "ssim": validation_summary.ssim,
                },
                step=epoch + 1,
            )

            # Sample outputs
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
                f"G_total: {avg_g_total:.4f} "
                f"D_loss: {avg_d_loss:.4f} "
                f"Val: {validation_summary.val_loss:.4f} "
                f"PSNR: {validation_summary.psnr:.4f} "
                f"SSIM: {validation_summary.ssim:.4f}"
            )

            # Checkpoint (generator only — D is auxiliary)
            checkpoint_path = output_dir / f"epoch_{epoch + 1}.pt"
            save_checkpoint(
                model=generator,
                optimizer=g_optimizer,
                epoch=epoch + 1,
                train_loss=avg_g_total,
                val_loss=validation_summary.val_loss,
                psnr=validation_summary.psnr,
                ssim=validation_summary.ssim,
                checkpoint_path=checkpoint_path,
            )

            # Best model tracking
            if validation_summary.val_loss < best_val_loss:
                best_val_loss = validation_summary.val_loss
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "generator_state_dict": generator.state_dict(),
                        "discriminator_state_dict": discriminator.state_dict(),
                        "g_optimizer_state_dict": g_optimizer.state_dict(),
                        "d_optimizer_state_dict": d_optimizer.state_dict(),
                        "train_loss": avg_g_total,
                        "val_loss": validation_summary.val_loss,
                        "psnr": validation_summary.psnr,
                        "ssim": validation_summary.ssim,
                    },
                    best_checkpoint_path,
                )
                print(f"Best model updated: {best_checkpoint_path}")
                mlflow.log_artifact(str(best_checkpoint_path))
                if cfg.mlflow.get("log_model", True):
                    mlflow.pytorch.log_model(generator, "best_generator")

            # Log artifacts
            if sample_dir is not None:
                mlflow.log_artifact(str(sample_dir), artifact_path="samples")
            mlflow.log_artifact(str(jsonl_path), artifact_path="metrics")

    return best_checkpoint_path


if __name__ == "__main__":
    train_gan()
