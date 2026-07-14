"""
DeepHorizon - Training Data Generator (512x512)
=================================================
Generates 10,000 clean/degraded pairs at 512x512 resolution.
Three model types: crescent (60%), ring (25%), double ring (15%).
Four degradation levels: light, medium, heavy, extreme.
Output: .npy float32 arrays (~20 GiB total: 1 MiB per image x 20,000 files).
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from pathlib import Path
import time

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "training"
CLEAN_DIR = OUTPUT_DIR / "clean"
DEGRADED_DIR = OUTPUT_DIR / "degraded"

NPIX = 512
FOV = 200.0  # micro-arcseconds
N_TOTAL = 10_000


def make_crescent(
    npix: int,
    fov: float,
    diameter: float,
    width: float,
    asymmetry: float,
    pa_deg: float,
    flux: float,
) -> np.ndarray:
    """Asymmetric crescent model (M87*-like)."""
    x = np.linspace(-fov / 2, fov / 2, npix)
    xx, yy = np.meshgrid(x, x)
    r = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)
    ring = np.exp(-0.5 * ((r - diameter / 2) / (width / 2)) ** 2)
    brightness = 1.0 + asymmetry * np.cos(theta - np.deg2rad(pa_deg))
    img = np.maximum(ring * brightness, 0)
    return img / img.sum() * flux


def make_ring(
    npix: int,
    fov: float,
    diameter: float,
    width: float,
    flux: float,
) -> np.ndarray:
    """Symmetric ring model."""
    x = np.linspace(-fov / 2, fov / 2, npix)
    xx, yy = np.meshgrid(x, x)
    r = np.sqrt(xx**2 + yy**2)
    ring = np.exp(-0.5 * ((r - diameter / 2) / (width / 2)) ** 2)
    return ring / ring.sum() * flux


def make_double_ring(
    npix: int,
    fov: float,
    d1: float,
    d2: float,
    width: float,
    flux: float,
) -> np.ndarray:
    """Double ring model (inner + outer, simulates jet structure)."""
    x = np.linspace(-fov / 2, fov / 2, npix)
    xx, yy = np.meshgrid(x, x)
    r = np.sqrt(xx**2 + yy**2)
    r1 = np.exp(-0.5 * ((r - d1 / 2) / (width / 2)) ** 2)
    r2 = np.exp(-0.5 * ((r - d2 / 2) / (width / 2)) ** 2) * 0.4
    img = r1 + r2
    return img / img.sum() * flux


def degrade(clean: np.ndarray, psf: float, noise: float, ds: int) -> np.ndarray:
    """Apply PSF blur, downsampling, and Gaussian noise."""
    out = gaussian_filter(clean, sigma=psf)
    if ds > 1:
        small = zoom(out, 1.0 / ds, order=1)
        out = zoom(small, ds, order=1)
        # Ensure shape matches after zoom rounding
        out = out[:clean.shape[0], :clean.shape[1]]
        if out.shape != clean.shape:
            padded = np.zeros_like(clean)
            padded[:out.shape[0], :out.shape[1]] = out
            out = padded
    if noise > 0:
        out = out + np.random.normal(0, noise * clean.max(), clean.shape)
    return np.clip(out, 0, None).astype(np.float32)


def main() -> None:
    for d in [CLEAN_DIR, DEGRADED_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)

    configs = [
        {"psf": 3.0,  "noise": 0.02, "ds": 1, "label": "light"},
        {"psf": 5.0,  "noise": 0.05, "ds": 2, "label": "medium"},
        {"psf": 8.0,  "noise": 0.10, "ds": 2, "label": "heavy"},
        {"psf": 12.0, "noise": 0.15, "ds": 4, "label": "extreme"},
    ]

    n_per_config = N_TOTAL // len(configs)  # 2500 each
    idx = 0
    t0 = time.time()

    for ci, cfg in enumerate(configs):
        label = cfg["label"]
        print(f"\n[{ci+1}/4] {label} — generating {n_per_config} pairs...")

        for i in range(n_per_config):
            diameter = rng.uniform(35, 50)
            width = rng.uniform(5, 14)
            flux = rng.uniform(0.5, 2.0)
            roll = rng.random()

            if roll < 0.60:
                # Crescent (60%)
                clean = make_crescent(
                    NPIX, FOV, diameter, width,
                    asymmetry=rng.uniform(0.1, 0.7),
                    pa_deg=rng.uniform(0, 360),
                    flux=flux,
                )
            elif roll < 0.85:
                # Ring (25%)
                clean = make_ring(NPIX, FOV, diameter, width, flux)
            else:
                # Double ring (15%)
                clean = make_double_ring(
                    NPIX, FOV,
                    d1=diameter,
                    d2=rng.uniform(60, 90),
                    width=width * 0.7,
                    flux=flux,
                )

            degraded = degrade(clean, cfg["psf"], cfg["noise"], cfg["ds"])

            np.save(CLEAN_DIR / f"{idx:05d}.npy", clean.astype(np.float32))
            np.save(DEGRADED_DIR / f"{idx:05d}.npy", degraded.astype(np.float32))
            idx += 1

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = idx / elapsed
                eta = (N_TOTAL - idx) / rate
                print(f"   {i+1}/{n_per_config} | total {idx}/{N_TOTAL} | {rate:.0f} pairs/s | ETA {eta:.0f}s")

    elapsed = time.time() - t0
    size_gb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*.npy")) / 1e9
    print(f"\n{'='*60}")
    print(f"  Done: {idx} pairs generated ({NPIX}x{NPIX})")
    print(f"  Size: {size_gb:.1f} GB")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  Dir:  {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
