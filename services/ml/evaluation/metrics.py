from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg, signal


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


# ---------------------------------------------------------------------------
# LPIPS — Learned Perceptual Image Patch Similarity
# Lazy-loaded global cache; lpips paketi ml.txt'de zaten var.
# ---------------------------------------------------------------------------
_LPIPS_CACHE: dict[str, torch.nn.Module] = {}


def _get_lpips_model(net: str = "alex", device: torch.device | None = None) -> torch.nn.Module:
    """Lazy-load and cache the LPIPS model."""
    import lpips  # type: ignore[import-untyped]

    key = f"{net}:{device}"
    if key not in _LPIPS_CACHE:
        model = lpips.LPIPS(net=net, verbose=False)
        if device is not None:
            model = model.to(device)
        model.eval()
        _LPIPS_CACHE[key] = model
    return _LPIPS_CACHE[key]


def compute_lpips(
    prediction: torch.Tensor,
    target: torch.Tensor,
    net: Literal["alex", "vgg", "squeeze"] = "alex",
) -> float:
    """Learned Perceptual Image Patch Similarity (lower = more similar).

    Args:
        prediction: Predicted image tensor (any shape, will be 4D).
        target: Target image tensor (same shape as prediction).
        net: Backbone network — "alex" (fast), "vgg" (accurate), "squeeze".

    Returns:
        LPIPS distance as a Python float. ~0.0 for identical images.
    """
    prediction, target = _validate_pair(prediction, target)
    device = prediction.device
    model = _get_lpips_model(net=net, device=device)

    # LPIPS expects input in [-1, 1] range
    prediction_norm = prediction * 2.0 - 1.0
    target_norm = target * 2.0 - 1.0

    with torch.no_grad():
        distance = model(prediction_norm, target_norm)

    return float(distance.mean().item())


# ---------------------------------------------------------------------------
# FID — Fréchet Inception Distance
# InceptionV3 pool_features (2048-dim) üzerinden iki dağılım arasındaki mesafe.
# ---------------------------------------------------------------------------
_INCEPTION_CACHE: dict[str, torch.nn.Module] = {}


def _get_inception_model(device: torch.device | None = None) -> torch.nn.Module:
    """Lazy-load and cache the InceptionV3 feature extractor."""
    from torchvision.models import Inception_V3_Weights, inception_v3  # type: ignore[import-untyped]

    key = str(device)
    if key not in _INCEPTION_CACHE:
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        # Remove final FC; use the 2048-dim feature before it
        model.fc = torch.nn.Identity()
        model.eval()
        if device is not None:
            model = model.to(device)
        _INCEPTION_CACHE[key] = model
    return _INCEPTION_CACHE[key]


def _inception_features(
    images: torch.Tensor,
    model: torch.nn.Module,
    batch_size: int = 32,
) -> torch.Tensor:
    """Extract 2048-dim features from InceptionV3 in batches."""
    features: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            batch = images[start : start + batch_size]
            # InceptionV3 expects 3-channel, 299x299, normalized
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=batch.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=batch.device).view(1, 3, 1, 1)
            batch = (batch - mean) / std
            feat = model(batch)
            features.append(feat.cpu())
    return torch.cat(features, dim=0).numpy()


def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    batch_size: int = 32,
) -> float:
    """Fréchet Inception Distance between two image sets (lower = more similar).

    Args:
        real_images: Real images tensor (N, C, H, W) in [0, 1].
        fake_images: Generated images tensor (M, C, H, W) in [0, 1].
        batch_size: InceptionV3 batch size for feature extraction.

    Returns:
        FID score as a Python float.
    """
    real_images = _ensure_4d(real_images).to(dtype=torch.float32)
    fake_images = _ensure_4d(fake_images).to(dtype=torch.float32)
    device = real_images.device
    model = _get_inception_model(device=device)

    real_features = _inception_features(real_images, model, batch_size=batch_size)
    fake_features = _inception_features(fake_images, model, batch_size=batch_size)

    mu_real = real_features.mean(axis=0)
    mu_fake = fake_features.mean(axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    diff = mu_real - mu_fake
    # sqrt(sigma_real @ sigma_fake) — bazen kompleks sonuç verir, real part al
    covmean = linalg.sqrtm(sigma_real @ sigma_fake)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_real) + np.trace(sigma_fake) - 2.0 * np.trace(covmean))
    return fid


# ---------------------------------------------------------------------------
# Physics-informed metrics — README'deki formel tanıma uygun
# ---------------------------------------------------------------------------
def compute_flux_conservation(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Flux conservation error: |sum(pred) - sum(target)| / sum(target).

    Toplam akı (flux) korunmalıdır; ideal değer 0.0.
    """
    prediction, target = _validate_pair(prediction, target)
    pred_sum = float(prediction.sum().item())
    target_sum = float(target.sum().item())

    if abs(target_sum) < 1e-12:
        return float("inf") if abs(pred_sum) > 1e-12 else 0.0

    return abs(pred_sum - target_sum) / abs(target_sum)


def compute_ring_diameter(
    image: torch.Tensor,
    n_bins: int = 360,
) -> float:
    """Ring diameter (pixels) via radial brightness profile peak detection.

    Görüntünün merkezinden dışa doğru radyal parlaklık profili çıkarılır,
    en parlak halka pikselinin yarıçapı döndürülür.

    Args:
        image: 2D, 3D veya 4D tensor (son iki boyut H, W).
        n_bins: Açısal bin sayısı (varsayılan 360 = 1 derece).

    Returns:
        Halka çapı (piksel cinsinden, çap = 2 * yarıçap).
    """
    image = _ensure_4d(image).to(dtype=torch.float32)
    # İlk batch ve kanalı al
    img = image[0, 0]  # (H, W)

    height, width = img.shape
    center_y, center_x = height / 2.0, width / 2.0
    max_radius = min(center_y, center_x)

    # Radyal profil: her yarıçap için ortalama parlaklık
    radii = torch.linspace(0, max_radius, n_bins + 1, device=img.device)
    profile = torch.zeros(n_bins, device=img.device)

    y_coords = torch.arange(height, device=img.device).float() - center_y
    x_coords = torch.arange(width, device=img.device).float() - center_x
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    rr = torch.sqrt(yy * yy + xx * xx)

    for i in range(n_bins):
        r_inner = radii[i].item()
        r_outer = radii[i + 1].item()
        mask = (rr >= r_inner) & (rr < r_outer)
        if mask.any():
            profile[i] = img[mask].mean()

    profile_np = profile.cpu().numpy()
    # En parlak piki bul (merkezden uzak, yarıçap > 5 piksel)
    peaks, _ = signal.find_peaks(profile_np, distance=5)
    if len(peaks) == 0:
        return 0.0

    # En yüksek piki seç
    best_peak = peaks[profile_np[peaks].argmax()]
    radius = float(radii[best_peak].item())
    return 2.0 * radius  # çap = 2 * yarıçap


def compute_asymmetry_ratio(
    image: torch.Tensor,
    ring_radius_px: float,
    n_bins: int = 360,
) -> float:
    """Brightness asymmetry ratio along the ring: max / min.

    Halka boyunca parlaklık asimetrisi; ideal değer 1.0 (simetrik).
    ring_radius_px verilmezse otomatik tespit edilir.
    """
    image = _ensure_4d(image).to(dtype=torch.float32)
    img = image[0, 0]  # (H, W)

    height, width = img.shape
    center_y, center_x = height / 2.0, width / 2.0

    if ring_radius_px <= 0:
        ring_radius_px = compute_ring_diameter(image, n_bins=n_bins) / 2.0

    # Halka üzerindeki pikselleri topla
    y_coords = torch.arange(height, device=img.device).float() - center_y
    x_coords = torch.arange(width, device=img.device).float() - center_x
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    rr = torch.sqrt(yy * yy + xx * xx)

    # Halka kalınlığı: ±2 piksel
    ring_mask = (rr >= ring_radius_px - 2.0) & (rr <= ring_radius_px + 2.0)
    if not ring_mask.any():
        return 1.0

    ring_values = img[ring_mask]
    min_val = float(ring_values.min().item())
    max_val = float(ring_values.max().item())

    if min_val < 1e-12:
        return float("inf") if max_val > 1e-12 else 1.0

    return max_val / min_val


def compute_physics_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Tüm physics-informed metrikleri tek seferde hesapla.

    Returns:
        Dict with keys: flux_error, ring_diameter_error, asymmetry_error.
    """
    prediction, target = _validate_pair(prediction, target)

    flux_error = compute_flux_conservation(prediction, target)

    pred_diameter = compute_ring_diameter(prediction)
    target_diameter = compute_ring_diameter(target)
    ring_diameter_error = abs(pred_diameter - target_diameter)

    pred_asym = compute_asymmetry_ratio(prediction, ring_radius_px=pred_diameter / 2.0)
    target_asym = compute_asymmetry_ratio(target, ring_radius_px=target_diameter / 2.0)
    asymmetry_error = abs(pred_asym - target_asym)

    return {
        "flux_error": flux_error,
        "ring_diameter_error": ring_diameter_error,
        "asymmetry_error": asymmetry_error,
    }


def compute_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    include_lpips: bool = False,
    include_physics: bool = False,
) -> dict[str, float]:
    """Tüm metrikleri tek seferde hesapla.

    Args:
        prediction: Predicted image tensor.
        target: Target image tensor.
        data_range: PSNR/SSIM için veri aralığı (varsayılan 1.0).
        include_lpips: True ise "lpips" anahtarı eklenir (yavaş, lazy load).
        include_physics: True ise flux/ring/asymmetry anahtarları eklenir.

    Returns:
        Dict with keys: psnr, ssim, [lpips], [flux_error, ring_diameter_error, asymmetry_error].
    """
    prediction, target = _validate_pair(prediction, target)
    metrics: dict[str, float] = {
        "psnr": compute_psnr(prediction, target, data_range=data_range),
        "ssim": compute_ssim(prediction, target, data_range=data_range),
    }
    if include_lpips:
        metrics["lpips"] = compute_lpips(prediction, target)
    if include_physics:
        metrics.update(compute_physics_metrics(prediction, target))
    return metrics
