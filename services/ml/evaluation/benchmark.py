from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from .metrics import compute_metrics


@dataclass(frozen=True)
class ValidationSummary:
    epoch: int
    train_loss: float
    val_loss: float
    psnr: float
    ssim: float
    lpips: float = 0.0
    flux_error: float = 0.0
    ring_diameter_error: float = 0.0
    asymmetry_error: float = 0.0


def _prepare_output_dir(output_dir: Path | str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_validation_summary(summary: ValidationSummary, output_dir: Path | str) -> Path:
    output_path = _prepare_output_dir(output_dir)
    results_path = output_path / "validation_results.jsonl"
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(summary), ensure_ascii=False) + "\n")
    return results_path


def save_sample_outputs(
    degraded: torch.Tensor,
    prediction: torch.Tensor,
    clean: torch.Tensor,
    output_dir: Path | str,
    epoch: int,
    max_samples: int = 4,
) -> Path | None:
    if degraded.numel() == 0:
        return None

    output_path = _prepare_output_dir(output_dir)
    sample_dir = output_path / "validation_samples" / f"epoch_{epoch:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_count = min(max_samples, degraded.shape[0])
    for index in range(sample_count):
        triplet = torch.cat(
            [
                degraded[index : index + 1],
                prediction[index : index + 1],
                clean[index : index + 1],
            ],
            dim=0,
        )
        grid = make_grid(triplet, nrow=3, normalize=True, value_range=(0.0, 1.0))
        save_image(grid, sample_dir / f"sample_{index:02d}.png")

    return sample_dir


def evaluate_validation_loader(
    model,
    val_loader,
    device: torch.device,
    criterion,
    prediction_transform: callable | None = None,
    data_range: float = 1.0,
) -> tuple[ValidationSummary, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    """Run validation over ``val_loader`` and aggregate metrics.

    Args:
        model: Generator / model under evaluation.
        val_loader: Validation DataLoader yielding ``(degraded, clean)`` pairs.
        device: Device to run inference on.
        criterion: Loss function used for ``val_loss`` (operates on raw
            model output).
        prediction_transform: Optional callable ``(pred) -> pred`` applied
            to the model output **before** metric computation. Pix2Pix
            generators with ``Tanh`` output ``[-1, 1]``; pass a transform
            that maps to ``[0, 1]`` so PSNR/SSIM are meaningful.
        data_range: ``data_range`` forwarded to ``compute_metrics``.
            Use ``2.0`` when comparing ``[-1, 1]`` predictions directly,
            or keep ``1.0`` after mapping to ``[0, 1]``.
    """
    model.eval()

    running_val_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    running_lpips = 0.0
    running_flux_error = 0.0
    running_ring_diameter_error = 0.0
    running_asymmetry_error = 0.0
    batch_count = 0
    sample_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    with torch.no_grad():
        for degraded, clean in val_loader:
            degraded = degraded.to(device, dtype=torch.float32)
            clean = clean.to(device, dtype=torch.float32)
            prediction = model(degraded)

            # Metrics are computed on the same range as the target.
            # Pix2Pix Tanh output is [-1, 1]; map to [0, 1] before metrics.
            metrics_input = (
                prediction_transform(prediction)
                if prediction_transform is not None
                else prediction
            )
            # val_loss is also computed on the mapped prediction so it
            # lives in the same range as the target (and as the
            # training-time pixel loss).
            loss = criterion(metrics_input, clean)
            metrics = compute_metrics(
                metrics_input,
                clean,
                data_range=data_range,
                include_lpips=True,
                include_physics=True,
            )

            running_val_loss += float(loss.item())
            running_psnr += metrics["psnr"]
            running_ssim += metrics["ssim"]
            running_lpips += metrics["lpips"]
            running_flux_error += metrics["flux_error"]
            running_ring_diameter_error += metrics["ring_diameter_error"]
            running_asymmetry_error += metrics["asymmetry_error"]
            batch_count += 1

            if sample_batch is None:
                sample_batch = (
                    degraded.detach().cpu(),
                    prediction.detach().cpu(),
                    clean.detach().cpu(),
                )

    if batch_count == 0:
        raise ValueError("val_loader produced no batches")

    summary = ValidationSummary(
        epoch=0,
        train_loss=0.0,
        val_loss=running_val_loss / batch_count,
        psnr=running_psnr / batch_count,
        ssim=running_ssim / batch_count,
        lpips=running_lpips / batch_count,
        flux_error=running_flux_error / batch_count,
        ring_diameter_error=running_ring_diameter_error / batch_count,
        asymmetry_error=running_asymmetry_error / batch_count,
    )
    return summary, sample_batch


def update_best_model(
    summary: ValidationSummary,
    model,
    optimizer,
    output_dir: Path | str,
    best_val_loss: float,
) -> tuple[float, bool, Path]:
    output_path = _prepare_output_dir(output_dir)
    best_checkpoint_path = output_path / "best_model.pt"

    if summary.val_loss >= best_val_loss:
        return best_val_loss, False, best_checkpoint_path

    torch.save(
        {
            "epoch": summary.epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": summary.train_loss,
            "val_loss": summary.val_loss,
            "psnr": summary.psnr,
            "ssim": summary.ssim,
        },
        best_checkpoint_path,
    )
    return summary.val_loss, True, best_checkpoint_path
