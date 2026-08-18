"""VGG-based perceptual loss for image restoration.

Uses a pretrained VGG19 feature extractor (ImageNet weights) and computes
the L1 distance between intermediate feature maps of the prediction and
target. This captures perceptual similarity rather than pixel-wise
agreement — small spatial shifts that look identical to humans still
yield high PSNR but low perceptual loss.

Reference:
    Johnson, J., Alahi, A., Fei-Fei, L. (2016).
    "Perceptual Losses for Real-Time Style Transfer and Super-Resolution."
    ECCV.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG19_Weights, vgg19  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Lazy-loaded VGG19 feature cache (avoid reloading weights per loss instance)
# ---------------------------------------------------------------------------
_VGG_CACHE: dict[str, torch.nn.Sequential] = {}


def _get_vgg_features(
    layer_name: str,
    device: torch.device | None = None,
) -> torch.nn.Sequential:
    """Return a frozen VGG19 feature extractor up to ``layer_name``.

    Args:
        layer_name: VGG19 layer to cut at. One of:
            - "relu1_2" (shallow, low-level features)
            - "relu2_2" (default, mid-level textures)
            - "relu3_3" (deeper, object parts)
            - "relu4_3" (very deep, semantic)
        device: Device to place the model on.

    Returns:
        A frozen ``nn.Sequential`` of VGG19 layers up to and including
        the requested activation.
    """
    key = f"{layer_name}:{device}"
    if key in _VGG_CACHE:
        return _VGG_CACHE[key]

    # VGG19 layer name → cut index in features
    cut_points: dict[str, int] = {
        "relu1_2": 4,   # after relu1_2
        "relu2_2": 9,   # after relu2_2
        "relu3_3": 18,  # after relu3_3
        "relu4_3": 27,  # after relu4_3
    }
    if layer_name not in cut_points:
        raise ValueError(
            f"Unknown VGG layer: {layer_name!r}. "
            f"Supported: {sorted(cut_points)}"
        )

    vgg = vgg19(weights=VGG19_Weights.DEFAULT)
    features = vgg.features[: cut_points[layer_name] + 1].eval()
    for param in features.parameters():
        param.requires_grad_(False)

    if device is not None:
        features = features.to(device)

    _VGG_CACHE[key] = features
    return features


class PerceptualLoss(nn.Module):
    """VGG19-based perceptual loss.

    Computes L1 distance between VGG19 feature maps of the prediction
    and target. Input is expected in ``[0, 1]`` range; the loss
    internally normalizes to ImageNet statistics.

    Args:
        layer: VGG19 layer to extract features from. Default ``"relu2_2"``
            (Johnson et al. recommendation for super-resolution).
        normalize: If ``True``, apply ImageNet mean/std normalization
            before feeding to VGG. Required for pretrained VGG to
            produce meaningful features.
        reduction: ``"mean"`` (scalar) or ``"sum"``.
    """

    def __init__(
        self,
        layer: Literal["relu1_2", "relu2_2", "relu3_3", "relu4_3"] = "relu2_2",
        normalize: bool = True,
        reduction: Literal["mean", "sum"] = "mean",
    ) -> None:
        super().__init__()

        if reduction not in {"mean", "sum"}:
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")

        self.layer = layer
        self.normalize = normalize
        self.reduction = reduction

        # ImageNet normalization constants
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        # VGG features are loaded lazily on first forward (need device).
        self._vgg: torch.nn.Sequential | None = None

    def _ensure_vgg(self, device: torch.device) -> None:
        if self._vgg is None:
            self._vgg = _get_vgg_features(self.layer, device=device)
            # Move buffers to the same device
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Convert grayscale → RGB and normalize to ImageNet stats."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if self.normalize:
            x = (x - self.mean) / self.std
        return x

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss between prediction and target.

        Args:
            prediction: Predicted image, shape ``(B, C, H, W)`` in ``[0, 1]``.
            target: Target image, same shape as prediction.

        Returns:
            Scalar loss tensor.
        """
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: prediction {tuple(prediction.shape)}"
                f" vs target {tuple(target.shape)}"
            )

        self._ensure_vgg(prediction.device)

        pred_features = self._vgg(self._preprocess(prediction))
        target_features = self._vgg(self._preprocess(target))

        if self.reduction == "mean":
            return F.l1_loss(pred_features, target_features)
        return F.l1_loss(pred_features, target_features, reduction="sum")
