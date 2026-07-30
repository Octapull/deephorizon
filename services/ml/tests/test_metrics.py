"""Unit tests for services.ml.evaluation.metrics."""

from __future__ import annotations

import pytest
import torch

from services.ml.evaluation.metrics import (
    compute_asymmetry_ratio,
    compute_fid,
    compute_flux_conservation,
    compute_lpips,
    compute_metrics,
    compute_physics_metrics,
    compute_psnr,
    compute_ring_diameter,
    compute_ssim,
)

# ---------- PSNR ----------


def test_compute_psnr_identical_returns_inf(sample_image: torch.Tensor) -> None:
    """Aynı görüntü → PSNR = inf."""
    assert compute_psnr(sample_image, sample_image) == float("inf")


def test_compute_psnr_different_returns_finite(
    sample_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Farklı görüntüler → PSNR sonlu ve negatif olmayan."""
    degraded, clean = sample_batch
    psnr = compute_psnr(degraded, clean)
    assert psnr != float("inf")
    assert psnr > 0.0


def test_compute_psnr_shape_mismatch_raises() -> None:
    """Farklı shape → ValueError."""
    a = torch.rand(1, 1, 64, 64)
    b = torch.rand(1, 1, 32, 32)
    with pytest.raises(ValueError, match="same shape"):
        compute_psnr(a, b)


# ---------- SSIM ----------


def test_compute_ssim_identical_high(sample_image: torch.Tensor) -> None:
    """Aynı görüntü → SSIM > 0.99."""
    ssim = compute_ssim(sample_image, sample_image)
    assert ssim > 0.99


def test_compute_ssim_different_lower(
    sample_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Farklı görüntüler → SSIM < 1.0."""
    degraded, clean = sample_batch
    ssim = compute_ssim(degraded, clean)
    assert 0.0 <= ssim < 1.0


# ---------- LPIPS ----------


def test_compute_lpips_identical_low(sample_image: torch.Tensor) -> None:
    """Aynı görüntü → LPIPS ≈ 0."""
    lpips = compute_lpips(sample_image, sample_image)
    assert lpips < 0.01


def test_compute_lpips_different_higher(
    sample_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Farklı görüntüler → LPIPS > 0."""
    degraded, clean = sample_batch
    lpips = compute_lpips(degraded, clean)
    assert lpips > 0.0


# ---------- FID ----------


def test_compute_fid_identical_low(
    sample_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Aynı görüntüler → FID ≈ 0."""
    _, clean = sample_batch
    fid = compute_fid(clean, clean)
    assert fid < 1.0


def test_compute_fid_different_higher(
    sample_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Farklı görüntüler → FID > 0."""
    degraded, clean = sample_batch
    fid = compute_fid(clean, degraded)
    assert fid > 0.0


# ---------- Physics: Flux Conservation ----------


def test_compute_flux_conservation_identical_zero(sample_image: torch.Tensor) -> None:
    """Aynı görüntü → flux error = 0."""
    err = compute_flux_conservation(sample_image, sample_image)
    assert err == 0.0


def test_compute_flux_conservation_different_positive(
    sample_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Farklı görüntüler → flux error > 0."""
    degraded, clean = sample_batch
    err = compute_flux_conservation(degraded, clean)
    assert err > 0.0


# ---------- Physics: Ring Diameter ----------


def test_compute_ring_diameter_returns_float(sample_image: torch.Tensor) -> None:
    """Ring diameter float döner."""
    diameter = compute_ring_diameter(sample_image)
    assert isinstance(diameter, float)


def test_compute_ring_diameter_zero_image() -> None:
    """Sıfır görüntü → diameter = 0 (peak yok)."""
    img = torch.zeros(1, 1, 64, 64)
    diameter = compute_ring_diameter(img)
    assert diameter == 0.0


# ---------- Physics: Asymmetry Ratio ----------


def test_compute_asymmetry_ratio_symmetric() -> None:
    """Simetrik görüntü → asymmetry ≈ 1.0 (max/min ≈ 1)."""
    img = torch.ones(1, 1, 64, 64)  # tüm pikseller 1 → halka üzerinde min=max
    ratio = compute_asymmetry_ratio(img, ring_radius_px=16.0)
    assert ratio == pytest.approx(1.0, abs=0.01)


def test_compute_asymmetry_ratio_asymmetric() -> None:
    """Asimetrik görüntü → asymmetry > 1.0 (max/min > 1)."""
    img = torch.ones(1, 1, 64, 64)
    # Sağ tarafta ekstra parlaklık ekle → halka üzerinde asimetri
    img[:, :, 16:48, 40:48] = 2.0  # sağa yaslı ekstra parlak bölge
    ratio = compute_asymmetry_ratio(img, ring_radius_px=16.0)
    assert ratio > 1.0


# ---------- Physics: Combined ----------


def test_compute_physics_metrics_keys(
    sample_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """3 physics metriği döner."""
    degraded, clean = sample_batch
    metrics = compute_physics_metrics(degraded, clean)
    assert "flux_error" in metrics
    assert "ring_diameter_error" in metrics
    assert "asymmetry_error" in metrics


# ---------- compute_metrics (combined) ----------


def test_compute_metrics_basic(sample_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Default: sadece psnr + ssim."""
    degraded, clean = sample_batch
    metrics = compute_metrics(degraded, clean)
    assert "psnr" in metrics
    assert "ssim" in metrics
    assert "lpips" not in metrics
    assert "flux_error" not in metrics


def test_compute_metrics_with_lpips(
    sample_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """include_lpips=True → lpips eklenir."""
    degraded, clean = sample_batch
    metrics = compute_metrics(degraded, clean, include_lpips=True)
    assert "lpips" in metrics


def test_compute_metrics_with_physics(
    sample_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """include_physics=True → 3 physics metriği eklenir."""
    degraded, clean = sample_batch
    metrics = compute_metrics(degraded, clean, include_physics=True)
    assert "flux_error" in metrics
    assert "ring_diameter_error" in metrics
    assert "asymmetry_error" in metrics
