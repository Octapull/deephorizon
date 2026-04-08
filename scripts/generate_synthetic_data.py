"""
DeepHorizon - Synthetic Black Hole Image Generator
====================================================
Generates clean/degraded training pairs using eht-imaging library.
Each pair: (degraded input, clean ground truth) at 128x128 resolution.
Suitable for rapid prototyping and model architecture experiments.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import ehtim as eh
    HAS_EHTIM = True
except ImportError:
    HAS_EHTIM = False
    print("  ehtim not found. Install with: pip install ehtim")

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "simulated"
CLEAN_DIR = OUTPUT_DIR / "clean"
DEGRADED_DIR = OUTPUT_DIR / "degraded"
PAIRS_DIR = OUTPUT_DIR / "pairs"


def create_crescent_model(
    total_flux: float = 1.0,
    fov_uas: float = 200.0,
    npix: int = 128,
    diameter_uas: float = 42.0,
    width_uas: float = 10.0,
    asymmetry: float = 0.3,
    pa_deg: float = 0.0,
) -> "eh.image.Image":
    """Generate a crescent (asymmetric ring) model resembling M87*."""
    fov_rad = fov_uas * eh.RADPERUAS

    im = eh.image.make_empty(npix, fov_rad, ra=187.7059308, dec=12.3911232, rf=230e9)

    x_arr = np.linspace(-fov_uas / 2, fov_uas / 2, npix)
    y_arr = np.linspace(-fov_uas / 2, fov_uas / 2, npix)
    xx, yy = np.meshgrid(x_arr, y_arr)

    r = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)

    ring = np.exp(-0.5 * ((r - diameter_uas / 2) / (width_uas / 2)) ** 2)
    pa_rad = np.deg2rad(pa_deg)
    brightness = 1.0 + asymmetry * np.cos(theta - pa_rad)
    crescent = ring * brightness
    crescent = np.maximum(crescent, 0)
    crescent = crescent / crescent.sum() * total_flux

    im.imvec = crescent.flatten()
    return im


def create_ring_model(
    total_flux: float = 1.0,
    fov_uas: float = 200.0,
    npix: int = 128,
    diameter_uas: float = 42.0,
    width_uas: float = 8.0,
) -> "eh.image.Image":
    """Generate a symmetric ring model."""
    fov_rad = fov_uas * eh.RADPERUAS
    im = eh.image.make_empty(npix, fov_rad, ra=187.7059308, dec=12.3911232, rf=230e9)

    x_arr = np.linspace(-fov_uas / 2, fov_uas / 2, npix)
    y_arr = np.linspace(-fov_uas / 2, fov_uas / 2, npix)
    xx, yy = np.meshgrid(x_arr, y_arr)
    r = np.sqrt(xx**2 + yy**2)

    ring = np.exp(-0.5 * ((r - diameter_uas / 2) / (width_uas / 2)) ** 2)
    ring = ring / ring.sum() * total_flux

    im.imvec = ring.flatten()
    return im


def add_gaussian_noise(image_array: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """Add Gaussian noise scaled to image peak intensity."""
    noise = np.random.normal(0, noise_level * image_array.max(), image_array.shape)
    return np.clip(image_array + noise, 0, None)


def apply_psf_blur(image_array: np.ndarray, sigma_pixels: float = 3.0) -> np.ndarray:
    """Apply Gaussian PSF convolution (simulates diffraction limit)."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(image_array, sigma=sigma_pixels)


def downsample(image_array: np.ndarray, factor: int = 2) -> np.ndarray:
    """Downsample and upsample back (simulates information loss)."""
    from scipy.ndimage import zoom
    small = zoom(image_array, 1.0 / factor, order=1)
    return zoom(small, factor, order=1)


def generate_degraded(clean: np.ndarray, config: dict) -> np.ndarray:
    """Apply physics-inspired degradation chain to a clean image."""
    degraded = clean.copy()

    if config.get("psf_blur", 0) > 0:
        degraded = apply_psf_blur(degraded, sigma_pixels=config["psf_blur"])

    if config.get("downsample_factor", 1) > 1:
        degraded = downsample(degraded, factor=config["downsample_factor"])

    if config.get("noise_level", 0) > 0:
        degraded = add_gaussian_noise(degraded, noise_level=config["noise_level"])

    return degraded


def save_pair(clean: np.ndarray, degraded: np.ndarray, idx: int, prefix: str) -> None:
    """Save clean/degraded pair as .npy files. Save PNG preview every 50 pairs."""
    clean_path = CLEAN_DIR / f"{prefix}_{idx:04d}.npy"
    degraded_path = DEGRADED_DIR / f"{prefix}_{idx:04d}.npy"

    np.save(clean_path, clean.astype(np.float32))
    np.save(degraded_path, degraded.astype(np.float32))

    if idx % 50 == 0:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(clean, cmap="afmhot", origin="lower")
        axes[0].set_title("Clean (Ground Truth)")
        axes[0].axis("off")
        axes[1].imshow(degraded, cmap="afmhot", origin="lower")
        axes[1].set_title("Degraded (Input)")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(PAIRS_DIR / f"{prefix}_{idx:04d}.png", dpi=100, bbox_inches="tight")
        plt.close()


def main() -> None:
    print("=" * 60)
    print("  DeepHorizon - Synthetic Data Generator (128x128)")
    print("=" * 60)

    if not HAS_EHTIM:
        print("\n  ehtim library required. Exiting.")
        return

    for d in [CLEAN_DIR, DEGRADED_DIR, PAIRS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # Degradation configs (light to extreme)
    degradation_configs = [
        {"psf_blur": 2.0, "noise_level": 0.02, "downsample_factor": 1, "label": "light"},
        {"psf_blur": 3.5, "noise_level": 0.05, "downsample_factor": 2, "label": "medium"},
        {"psf_blur": 5.0, "noise_level": 0.10, "downsample_factor": 2, "label": "heavy"},
        {"psf_blur": 7.0, "noise_level": 0.15, "downsample_factor": 4, "label": "extreme"},
    ]

    n_per_config = 250  # 250 per config -> 1000 total pairs
    total = 0

    for config in degradation_configs:
        label = config["label"]
        print(f"\n  Degradation: {label}")
        print(f"   PSF blur={config['psf_blur']}, noise={config['noise_level']}, downsample={config['downsample_factor']}x")

        for i in range(n_per_config):
            # Randomize physical parameters
            diameter = np.random.uniform(35, 50)  # uas
            width = np.random.uniform(6, 14)
            asymmetry = np.random.uniform(0.1, 0.6)
            pa = np.random.uniform(0, 360)
            flux = np.random.uniform(0.5, 2.0)

            # Select model (70% crescent, 30% ring)
            if np.random.random() < 0.7:
                im = create_crescent_model(
                    total_flux=flux,
                    diameter_uas=diameter,
                    width_uas=width,
                    asymmetry=asymmetry,
                    pa_deg=pa,
                )
                prefix = f"crescent_{label}"
            else:
                im = create_ring_model(
                    total_flux=flux,
                    diameter_uas=diameter,
                    width_uas=width,
                )
                prefix = f"ring_{label}"

            clean = im.imvec.reshape(im.ydim, im.xdim)
            degraded = generate_degraded(clean, config)

            save_pair(clean, degraded, total, prefix)
            total += 1

            if (i + 1) % 100 == 0:
                print(f"   [{i + 1}/{n_per_config}] generated")

    print(f"\n{'=' * 60}")
    print(f"  Done: {total} training pairs generated")
    print(f"  Clean:    {CLEAN_DIR}")
    print(f"  Degraded: {DEGRADED_DIR}")
    print(f"  Previews: {PAIRS_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
