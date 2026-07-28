"""Physics-informed loss for black hole image reconstruction.

Combines three physically-motivated penalties that encourage the
generator to produce images consistent with the underlying physics:

1. **Flux conservation**: Total flux (sum of pixel values) should be
   preserved between prediction and target.
2. **Ring diameter**: The diameter of the brightest ring should match
   the target's ring diameter.
3. **Asymmetry**: The brightness asymmetry ratio along the ring should
   match the target's asymmetry.

All three are differentiable approximations of the metrics in
``services.ml.evaluation.metrics``. The ring/asymmetry components use
a soft radial profile (Gaussian-weighted) to keep gradients flowing.

Reference:
    See ``docs/SOZLUK.md`` for the formal definitions of flux, ring
    diameter, and asymmetry in the EHT context.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsLoss(nn.Module):
    """Physics-informed loss combining flux, ring, and asymmetry terms.

    Args:
        flux_weight: Weight for the flux conservation term.
        ring_weight: Weight for the ring diameter term.
        asymmetry_weight: Weight for the asymmetry term.
        n_bins: Number of angular bins for the radial profile.
        sigma: Gaussian sigma (in pixels) for soft radial binning.
    """

    def __init__(
        self,
        flux_weight: float = 1.0,
        ring_weight: float = 1.0,
        asymmetry_weight: float = 1.0,
        n_bins: int = 64,
        sigma: float = 2.0,
    ) -> None:
        super().__init__()

        if flux_weight < 0 or ring_weight < 0 or asymmetry_weight < 0:
            raise ValueError("All weights must be non-negative")

        self.flux_weight = flux_weight
        self.ring_weight = ring_weight
        self.asymmetry_weight = asymmetry_weight
        self.n_bins = n_bins
        self.sigma = sigma

    # ------------------------------------------------------------------
    # Differentiable radial profile
    # ------------------------------------------------------------------
    def _radial_profile(
        self,
        image: torch.Tensor,
        n_bins: int,
    ) -> torch.Tensor:
        """Compute a soft radial brightness profile.

        Args:
            image: 4D tensor ``(B, 1, H, W)``.
            n_bins: Number of radial bins.

        Returns:
            Profile tensor of shape ``(B, n_bins)``.
        """
        batch_size, _, height, width = image.shape
        device = image.device
        dtype = image.dtype

        center_y = (height - 1) / 2.0
        center_x = (width - 1) / 2.0
        max_radius = min(center_y, center_x)

        # Radial coordinate grid
        y = torch.arange(height, device=device, dtype=dtype) - center_y
        x = torch.arange(width, device=device, dtype=dtype) - center_x
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        rr = torch.sqrt(yy * yy + xx * xx)  # (H, W)

        # Bin centers
        bin_edges = torch.linspace(0, max_radius, n_bins + 1, device=device, dtype=dtype)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # (n_bins,)

        # Soft assignment via Gaussian weighting
        # rr: (H, W) → (1, H, W); bin_centers: (n_bins,) → (n_bins, 1, 1)
        diff = rr.unsqueeze(0) - bin_centers.view(-1, 1, 1)  # (n_bins, H, W)
        weights = torch.exp(-(diff ** 2) / (2.0 * self.sigma ** 2))  # (n_bins, H, W)

        # Normalize weights per bin
        weights = weights / weights.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)

        # Weighted average per bin
        img_flat = image[:, 0]  # (B, H, W)
        profile = torch.einsum("bhw,nhw->bn", img_flat, weights)  # (B, n_bins)

        return profile

    def _ring_diameter_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Soft ring diameter loss via profile peak alignment.

        Finds the peak of the radial profile (excluding the central
        DC component) and penalizes the absolute difference between
        prediction and target peak radii.
        """
        pred_profile = self._radial_profile(prediction, self.n_bins)
        target_profile = self._radial_profile(target, self.n_bins)

        # Skip the first 2 bins (central DC component)
        pred_profile = pred_profile[:, 2:]
        target_profile = target_profile[:, 2:]

        # Soft argmax via weighted average (differentiable)
        bin_indices = torch.arange(
            pred_profile.shape[1], device=prediction.device, dtype=prediction.dtype
        )
        pred_peak = (pred_profile * bin_indices).sum(dim=1) / pred_profile.sum(dim=1).clamp_min(1e-12)
        target_peak = (target_profile * bin_indices).sum(dim=1) / target_profile.sum(dim=1).clamp_min(1e-12)

        return F.l1_loss(pred_peak, target_peak)

    def _asymmetry_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Asymmetry loss: variance of the radial profile.

        A perfectly symmetric ring has zero variance in its radial
        profile (after peak normalization). We use the profile
        variance as a proxy for asymmetry.
        """
        pred_profile = self._radial_profile(prediction, self.n_bins)
        target_profile = self._radial_profile(target, self.n_bins)

        # Normalize profiles to [0, 1] for scale invariance
        pred_norm = pred_profile / pred_profile.sum(dim=1, keepdim=True).clamp_min(1e-12)
        target_norm = target_profile / target_profile.sum(dim=1, keepdim=True).clamp_min(1e-12)

        # Variance as asymmetry proxy
        pred_var = pred_norm.var(dim=1)
        target_var = target_norm.var(dim=1)

        return F.l1_loss(pred_var, target_var)

    def _flux_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Flux conservation loss: relative error of total flux."""
        pred_sum = prediction.sum(dim=(1, 2, 3))
        target_sum = target.sum(dim=(1, 2, 3))

        # Avoid division by zero
        safe_target = target_sum.abs().clamp_min(1e-6)
        rel_error = (pred_sum - target_sum).abs() / safe_target

        return rel_error.mean()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the weighted physics loss.

        Args:
            prediction: Predicted image, shape ``(B, 1, H, W)`` in ``[0, 1]``.
            target: Target image, same shape as prediction.

        Returns:
            Scalar loss tensor.
        """
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: prediction {tuple(prediction.shape)}"
                f" vs target {tuple(target.shape)}"
            )

        flux = self._flux_loss(prediction, target)
        ring = self._ring_diameter_loss(prediction, target)
        asym = self._asymmetry_loss(prediction, target)

        total = (
            self.flux_weight * flux
            + self.ring_weight * ring
            + self.asymmetry_weight * asym
        )
        return total
