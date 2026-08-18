import torch.nn as nn

# Faz 1 Adım 6: mse | l1 | smooth_l1
# Faz 2 Adım 12: perceptual, gan_g, gan_d, physics, combined
_SUPPORTED_LOSSES = frozenset(
    {"mse", "l1", "smooth_l1", "perceptual", "gan_g", "gan_d", "physics", "combined"}
)


def get_loss(name: str = "mse", **kwargs) -> nn.Module:
    """Factory for loss functions.

    Args:
        name: Loss identifier. One of:
            - ``"mse"``, ``"l1"``, ``"smooth_l1"``: pixel losses.
            - ``"perceptual"``: VGG19 perceptual loss.
            - ``"gan_g"``: generator adversarial loss (vanilla/lsgan).
            - ``"gan_d"``: discriminator adversarial loss (vanilla/lsgan).
            - ``"physics"``: physics-informed loss (flux + ring + asymmetry).
            - ``"combined"``: weighted sum of pixel + perceptual + adv + physics.
        **kwargs: Extra arguments forwarded to the loss constructor.

    Returns:
        An instantiated ``nn.Module`` loss.

    Raises:
        ValueError: If ``name`` is not a supported loss identifier.
    """
    # Pixel losses (Faz 1)
    if name == "mse":
        return nn.MSELoss(**kwargs)
    if name == "l1":
        return nn.L1Loss(**kwargs)
    if name == "smooth_l1":
        return nn.SmoothL1Loss(**kwargs)

    # Faz 2 losses — lazy import to avoid loading VGG unless needed
    if name == "perceptual":
        from services.ml.losses.perceptual import PerceptualLoss

        return PerceptualLoss(**kwargs)

    if name == "gan_g":
        from services.ml.losses.gan import GeneratorAdversarialLoss

        return GeneratorAdversarialLoss(**kwargs)

    if name == "gan_d":
        from services.ml.losses.gan import DiscriminatorAdversarialLoss

        return DiscriminatorAdversarialLoss(**kwargs)

    if name == "physics":
        from services.ml.losses.physics import PhysicsLoss

        return PhysicsLoss(**kwargs)

    if name == "combined":
        from services.ml.losses.combined import CombinedLoss

        return CombinedLoss(**kwargs)

    supported = ", ".join(sorted(_SUPPORTED_LOSSES))
    raise ValueError(
        f"Unknown loss: {name!r}. Supported losses: {supported}."
    )
