"""ESRGAN training loop for super-resolution image enhancement.

Trains an ESRGANGenerator and RaDiscriminator in alternating fashion
using the CombinedLoss objective. Supports:

- Mixed precision (AMP) with BF16/FP16
- Gradient accumulation
- MLflow experiment tracking
- Hydra configuration
- Checkpoint save/load for both G and D
- Best-model tracking by validation loss
- Pix2Pix warm-start (scale=1 modunda)

The training follows the ESRGAN recipe:
    1. Forward pass: fake = G(condition)
    2. Discriminator step: minimize D loss on (real, fake) — RaGAN
    3. Generator step: minimize G loss (pixel + perceptual + adv + physics)
    4. Log metrics, save checkpoint, repeat

Referans:
    Wang, X. et al. (2018). "ESRGAN: Enhanced Super-Resolution
    Generative Adversarial Networks." ECCV Workshops.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import hydra
import mlflow
import mlflow.pytorch
import torch
from omegaconf import DictConfig, OmegaConf

# PyTorch 2.2.x uyumluluğu: torch.amp.GradScaler 2.3+ ile gelir.
try:
    from torch.amp import GradScaler, autocast as _torch_autocast

    _AMP_HAS_DTYPE = True
except ImportError:  # pragma: no cover - eski torch sürümleri için
    from torch.cuda.amp import GradScaler, autocast as _torch_autocast

    _AMP_HAS_DTYPE = False


def _autocast_ctx(enabled: bool, amp_dtype: torch.dtype):
    """PyTorch 2.2/2.3+ uyumlu autocast context manager."""
    if not enabled:
        return _torch_autocast(enabled=False)
    if _AMP_HAS_DTYPE:
        return _torch_autocast("cuda", enabled=True, dtype=amp_dtype)
    return _torch_autocast(enabled=True)


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
from services.ml.models.esrgan import ESRGANGenerator, RaDiscriminator


def _build_generator(cfg: DictConfig, device: torch.device) -> ESRGANGenerator:
    """ESRGAN generator'ü config'den oluştur."""
    gen_cfg = cfg.model.generator
    generator = ESRGANGenerator(
        in_channels=int(getattr(cfg.model, "in_channels", 1)),
        out_channels=int(getattr(cfg.model, "out_channels", 1)),
        num_rrdb=int(getattr(gen_cfg, "num_rrdb", 23)),
        features=int(getattr(gen_cfg, "features", 64)),
        growth_rate=int(getattr(gen_cfg, "growth_rate", 32)),
        scale=int(getattr(gen_cfg, "scale", 1)),
        use_tanh=bool(getattr(gen_cfg, "use_tanh", True)),
    )
    return generator.to(device)


def _build_discriminator(cfg: DictConfig, device: torch.device) -> RaDiscriminator:
    """RaGAN discriminator'ü config'den oluştur."""
    disc_cfg = getattr(cfg.model, "discriminator", None)
    if disc_cfg is None:
        disc_cfg = SimpleNamespace(
            in_channels=2,
            base_channels=64,
            max_channels=512,
            n_layers=5,
        )
    discriminator = RaDiscriminator(
        in_channels=int(getattr(disc_cfg, "in_channels", 2)),
        base_channels=int(getattr(disc_cfg, "base_channels", 64)),
        max_channels=int(getattr(disc_cfg, "max_channels", 512)),
        n_layers=int(getattr(disc_cfg, "n_layers", 5)),
    )
    return discriminator.to(device)


def _build_losses(cfg: DictConfig) -> tuple[CombinedLoss, DiscriminatorAdversarialLoss]:
    """Generator (combined) ve discriminator loss'larını oluştur."""
    loss_cfg = cfg.loss
    weights = loss_cfg.weights

    generator_loss = CombinedLoss(
        pixel_weight=float(getattr(weights, "pixel", 100.0)),
        perceptual_weight=float(getattr(weights, "perceptual", 1.0)),
        adversarial_weight=float(getattr(weights, "adversarial", 1.0)),
        physics_weight=float(getattr(weights, "physics", 0.5)),
        pixel_loss=str(getattr(loss_cfg, "pixel_loss", "l1")),
        gan_mode=str(getattr(loss_cfg, "gan_mode", "lsgan")),
        perceptual_layer=str(getattr(loss_cfg, "perceptual_layer", "relu2_2")),
    )

    discriminator_loss = DiscriminatorAdversarialLoss(
        mode=str(getattr(loss_cfg, "gan_mode", "lsgan")),
    )

    return generator_loss, discriminator_loss


def _build_optimizers(
    cfg: DictConfig,
    generator: ESRGANGenerator,
    discriminator: RaDiscriminator,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """G ve D için Adam optimizer'lar oluştur."""
    opt_cfg = cfg.training.optimizer
    lr = float(cfg.training.learning_rate)
    betas = tuple(opt_cfg.betas)
    weight_decay = float(opt_cfg.weight_decay)

    d_lr = lr * float(getattr(cfg.training, "discriminator_lr_factor", 0.5))

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
    discriminator: RaDiscriminator,
    d_optimizer: torch.optim.Optimizer,
    d_loss_fn: DiscriminatorAdversarialLoss,
    condition: torch.Tensor,
    real_target: torch.Tensor,
    fake_target: torch.Tensor,
    use_amp: bool,
    amp_dtype: torch.dtype,
    scaler: GradScaler | None,
) -> tuple[float, torch.Tensor]:
    """Tek discriminator adımı — RaGAN relativistic logit'lerle.

    Returns:
        Tuple of ``(d_loss, disc_real)``. ``disc_real`` G step'inde
        tekrar hesaplanmaması için saklanır.
    """
    discriminator.train()
    d_optimizer.zero_grad()

    with _autocast_ctx(use_amp, amp_dtype):
        disc_real = discriminator(condition, real_target)
        disc_fake = discriminator(condition, fake_target)
        # RaGAN: relativistic logit'ler
        D_real_rel, D_fake_rel = discriminator.relativistic_logits(disc_real, disc_fake)
        d_loss = d_loss_fn(D_real_rel, D_fake_rel)

    if use_amp and amp_dtype == torch.float16 and scaler is not None:
        scaler.scale(d_loss).backward()
        scaler.step(d_optimizer)
        scaler.update()
    else:
        d_loss.backward()
        d_optimizer.step()

    return float(d_loss.item()), disc_real.detach()


def _train_generator_step(
    generator: ESRGANGenerator,
    discriminator: RaDiscriminator,
    g_optimizer: torch.optim.Optimizer,
    g_loss_fn: CombinedLoss,
    condition: torch.Tensor,
    real_target: torch.Tensor,
    disc_real: torch.Tensor,
    use_amp: bool,
    amp_dtype: torch.dtype,
    scaler: GradScaler | None,
) -> dict[str, float]:
    """Tek generator adımı — RaGAN + combined loss.

    Args:
        disc_real: D step'inde hesaplanan ``discriminator(condition, real_target)``
            çıktısı. G step'inde tekrar hesaplanmaz (verimlilik).
    """
    generator.train()
    g_optimizer.zero_grad()

    with _autocast_ctx(use_amp, amp_dtype):
        fake_target = generator(condition)
        disc_fake = discriminator(condition, fake_target)
        # RaGAN: fake için relativistic logit (disc_real D step'ten reuse)
        _, D_fake_rel = discriminator.relativistic_logits(disc_real, disc_fake)
        components = g_loss_fn(fake_target, real_target, disc_output=D_fake_rel)
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
def train_esrgan(cfg: DictConfig) -> Path:
    """ESRGAN eğitimi — black hole görüntü çiftleri üzerinde.

    Tüm hiperparametreler Hydra config'den gelir. CLI override:
        python -m services.ml.training.train_esrgan training.epochs=2

    Her run MLflow'a loglanır: params (full config), per-epoch metrics
    (G loss, D loss, PSNR, SSIM), artifacts (best_model.pt, sample PNGs).
    """
    # Cihaz
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    # Çıktı dizini
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    torch.manual_seed(cfg.seed)

    # Veri
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

    # Modeller
    generator = _build_generator(cfg, device)
    discriminator = _build_discriminator(cfg, device)

    # Warm-start: Pix2Pix checkpoint'ından transfer learning (opsiyonel)
    warm_start_from = cfg.training.get("warm_start_from")
    if warm_start_from:
        warm_start_path = Path(warm_start_from)
        if warm_start_path.exists():
            print(f"Warm-start: Pix2Pix checkpoint yükleniyor → {warm_start_path}")
            # ESRGANGenerator.from_pix2pix_checkpoint() — kanal çıkarımı + mimari kurulum
            warm_started = ESRGANGenerator.from_pix2pix_checkpoint(
                checkpoint_path=str(warm_start_path),
                scale=cfg.model.generator.scale,
                device=device,
            )
            # Ağırlıkları mevcut generator'a aktar (strict=False — mimari farklı)
            missing, unexpected = generator.load_state_dict(
                warm_started.state_dict(), strict=False
            )
            print(f"  Loaded: {len(warm_started.state_dict())} keys")
            print(f"  Missing (yeni katmanlar): {len(missing)} keys")
            print(f"  Unexpected (Pix2Pix-only): {len(unexpected)} keys")
        else:
            print(f"Warm-start checkpoint bulunamadı: {warm_start_path}")

    # Loss'lar
    generator_loss, discriminator_loss = _build_losses(cfg)

    # Optimizer'lar
    g_optimizer, d_optimizer = _build_optimizers(cfg, generator, discriminator)

    # AMP
    use_amp = bool(cfg.training.amp) and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16 if cfg.training.amp_dtype == "bfloat16" else torch.float16
    )
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
                # D için fake üret (grad yok)
                generator.eval()
                with torch.no_grad():
                    fake_for_d = generator(degraded)
                generator.train()

                # D adımı
                d_loss, disc_real = _train_discriminator_step(
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

                # G adımı (disc_real D step'ten reuse — verimlilik)
                g_components = _train_generator_step(
                    generator=generator,
                    discriminator=discriminator,
                    g_optimizer=g_optimizer,
                    g_loss_fn=generator_loss,
                    condition=degraded,
                    real_target=clean,
                    disc_real=disc_real,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    scaler=scaler,
                )

                # Metrik biriktir
                running_g_total += g_components["total"]
                running_g_pixel += g_components.get("pixel", 0.0)
                running_g_perceptual += g_components.get("perceptual", 0.0)
                running_g_adversarial += g_components.get("adversarial", 0.0)
                running_g_physics += g_components.get("physics", 0.0)
                running_d_loss += d_loss
                batch_count += 1

            # Ortalama metrikler
            avg_g_total = running_g_total / max(1, batch_count)
            avg_g_pixel = running_g_pixel / max(1, batch_count)
            avg_g_perceptual = running_g_perceptual / max(1, batch_count)
            avg_g_adversarial = running_g_adversarial / max(1, batch_count)
            avg_g_physics = running_g_physics / max(1, batch_count)
            avg_d_loss = running_d_loss / max(1, batch_count)

            # ---- Validation ----
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

            # Checkpoint (generator only)
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
                    mlflow.pytorch.log_model(
                        generator,
                        name="best_generator",
                        input_example=degraded[:1].detach().cpu().numpy(),
                        serialization_format="pickle",
                    )

            # Log artifacts
            if sample_dir is not None:
                mlflow.log_artifact(str(sample_dir), artifact_path="samples")
            mlflow.log_artifact(str(jsonl_path), artifact_path="metrics")

    return best_checkpoint_path


if __name__ == "__main__":
    train_esrgan()
