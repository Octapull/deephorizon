import torch.nn as nn

# Faz 1 Adım 6: mse | l1 | smooth_l1
# Faz 2'de eklenecek: perceptual, gan, physics, combined
_SUPPORTED_LOSSES = frozenset({"mse", "l1", "smooth_l1"})


def get_loss(name: str = "mse", **kwargs) -> nn.Module:
    """Factory for loss functions.

    Args:
        name: Loss identifier. One of "mse", "l1", "smooth_l1".
        **kwargs: Extra arguments forwarded to the loss constructor
            (e.g. beta for SmoothL1Loss, reduction for any of them).

    Returns:
        An instantiated `nn.Module` loss.

    Raises:
        ValueError: If `name` is not a supported loss identifier.

    Note:
        Faz 2'de eklenecek: "perceptual" (VGG), "gan" (adversarial),
        "physics" (flux + ring + asymmetry), "combined" (weighted sum).
    """
    if name == "mse":
        return nn.MSELoss(**kwargs)
    if name == "l1":
        return nn.L1Loss(**kwargs)
    if name == "smooth_l1":
        return nn.SmoothL1Loss(**kwargs)

    supported = ", ".join(sorted(_SUPPORTED_LOSSES))
    raise ValueError(
        f"Unknown loss: {name!r}. Supported losses: {supported}. "
        f"physics/perceptual/gan will be added in Faz 2."
    )
