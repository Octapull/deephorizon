from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected a 2D, 3D, or 4D tensor, got shape {tuple(tensor.shape)}")
    return tensor


def _validate_pair(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = _ensure_4d(prediction).to(dtype=torch.float32)
    target = _ensure_4d(target).to(dtype=torch.float32)

    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape, "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )

    return prediction, target


def compute_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    prediction, target = _validate_pair(prediction, target)
    mse = F.mse_loss(prediction, target, reduction="mean")

    if mse <= 0:
        return float("inf")

    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(float(mse)))


def _gaussian_kernel(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinates = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gaussian = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    window_2d = gaussian[:, None] * gaussian[None, :]
    window_2d = window_2d / window_2d.sum()
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def compute_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    prediction, target = _validate_pair(prediction, target)

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    channels = prediction.shape[1]
    kernel = _gaussian_kernel(window_size, sigma, channels, prediction.device, prediction.dtype)
    padding = window_size // 2

    mu_prediction = F.conv2d(prediction, kernel, padding=padding, groups=channels)
    mu_target = F.conv2d(target, kernel, padding=padding, groups=channels)

    mu_prediction_sq = mu_prediction.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_prediction_target = mu_prediction * mu_target

    sigma_prediction_sq = F.conv2d(prediction * prediction, kernel, padding=padding, groups=channels) - mu_prediction_sq
    sigma_target_sq = F.conv2d(target * target, kernel, padding=padding, groups=channels) - mu_target_sq
    sigma_prediction_target = (
        F.conv2d(prediction * target, kernel, padding=padding, groups=channels) - mu_prediction_target
    )

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2.0 * mu_prediction_target + c1) * (2.0 * sigma_prediction_target + c2)
    denominator = (mu_prediction_sq + mu_target_sq + c1) * (sigma_prediction_sq + sigma_target_sq + c2)

    ssim_map = numerator / denominator.clamp_min(1e-12)
    return float(ssim_map.mean().item())


def compute_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> dict[str, float]:
    prediction, target = _validate_pair(prediction, target)
    return {
        "psnr": compute_psnr(prediction, target, data_range=data_range),
        "ssim": compute_ssim(prediction, target, data_range=data_range),
    }
