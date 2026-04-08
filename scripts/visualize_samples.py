"""
DeepHorizon - Data Visualization
==================================
Renders EHT real observations as dirty images and generates
high-quality PNG comparisons for synthetic clean/degraded pairs.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "visualizations"
SIMULATED_DIR = Path(__file__).parent.parent / "data" / "raw" / "simulated"
EHT_DIR = Path(__file__).parent.parent / "data" / "raw" / "eht"


def plot_style() -> None:
    """Set dark theme for all plots."""
    plt.rcParams.update({
        "figure.facecolor": "#0a0a0a",
        "axes.facecolor": "#0a0a0a",
        "text.color": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#888",
        "ytick.color": "#888",
        "font.family": "sans-serif",
        "font.size": 11,
    })


def save_single_image(data: np.ndarray, path: Path, title: str = "", cmap: str = "afmhot") -> None:
    """Save a single image as high-quality PNG."""
    plot_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(data, cmap=cmap, origin="lower", norm=PowerNorm(gamma=0.5))
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout(pad=0.5)
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close()


def save_comparison(clean: np.ndarray, degraded: np.ndarray, path: Path, label: str = "") -> None:
    """Save side-by-side degraded vs clean comparison."""
    plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    norm = PowerNorm(gamma=0.5, vmin=0, vmax=clean.max())

    axes[0].imshow(degraded, cmap="afmhot", origin="lower", norm=PowerNorm(gamma=0.5))
    axes[0].set_title("Degraded Input", fontsize=13, fontweight="bold", pad=10)
    axes[0].axis("off")

    axes[1].imshow(clean, cmap="afmhot", origin="lower", norm=norm)
    axes[1].set_title("Clean Target (Ground Truth)", fontsize=13, fontweight="bold", pad=10)
    axes[1].axis("off")

    if label:
        fig.suptitle(label, fontsize=16, fontweight="bold", y=1.02, color="#FF6B35")

    plt.tight_layout(pad=1.0)
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close()


def save_grid(pairs: list, path: Path, title: str = "") -> None:
    """Save Nx2 grid: top row degraded, bottom row clean."""
    plot_style()
    n = len(pairs)
    fig, axes = plt.subplots(2, max(n, 2), figsize=(4 * max(n, 2), 8))
    if n == 1:
        axes = axes.reshape(2, -1)

    for i, (clean, degraded, label) in enumerate(pairs):
        norm = PowerNorm(gamma=0.5, vmin=0, vmax=clean.max())

        axes[0, i].imshow(degraded, cmap="afmhot", origin="lower", norm=PowerNorm(gamma=0.5))
        axes[0, i].set_title(f"Degraded\n({label})", fontsize=10, fontweight="bold")
        axes[0, i].axis("off")

        axes[1, i].imshow(clean, cmap="afmhot", origin="lower", norm=norm)
        axes[1, i].set_title("Clean", fontsize=10, fontweight="bold")
        axes[1, i].axis("off")

    if title:
        fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02, color="#FF6B35")

    plt.tight_layout(pad=0.8)
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close()


def visualize_eht_data() -> None:
    """Render EHT UVFITS observations as dirty images."""
    try:
        import ehtim as eh
    except ImportError:
        print("  ehtim not installed, skipping EHT visualization")
        return

    eht_vis_dir = OUTPUT_DIR / "eht"
    eht_vis_dir.mkdir(parents=True, exist_ok=True)

    uvfits_files = sorted(EHT_DIR.rglob("*.uvfits"))
    if not uvfits_files:
        print("  No UVFITS files found")
        return

    print(f"\n  EHT Real Data Visualization ({len(uvfits_files)} files)")

    for uvf in uvfits_files:
        try:
            obs = eh.obsdata.load_uvfits(str(uvf))
            im = obs.dirtyimage(128, 200 * eh.RADPERUAS)
            data = im.imvec.reshape(im.ydim, im.xdim)

            name = uvf.stem
            save_single_image(
                data,
                eht_vis_dir / f"{name}.png",
                title=f"EHT Dirty Image\n{name}",
            )
            print(f"  ok {name}.png")
        except Exception as e:
            print(f"  FAILED {uvf.name}: {e}")

    # All low-band observations in one grid
    try:
        lo_files = sorted((EHT_DIR / "m87_2017").glob("*_lo_*.uvfits"))
        if len(lo_files) >= 4:
            plot_style()
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            days = ["Day 095", "Day 096", "Day 100", "Day 101"]
            for i, uvf in enumerate(lo_files[:4]):
                obs = eh.obsdata.load_uvfits(str(uvf))
                im = obs.dirtyimage(128, 200 * eh.RADPERUAS)
                data = im.imvec.reshape(im.ydim, im.xdim)
                axes[i].imshow(data, cmap="afmhot", origin="lower", norm=PowerNorm(gamma=0.5))
                axes[i].set_title(days[i], fontsize=13, fontweight="bold")
                axes[i].axis("off")
            fig.suptitle("M87* — EHT 2017 Low Band Observations (Dirty Images)",
                         fontsize=16, fontweight="bold", y=1.02, color="#FF6B35")
            plt.tight_layout(pad=0.8)
            plt.savefig(eht_vis_dir / "m87_2017_lo_grid.png", dpi=200, bbox_inches="tight", facecolor="#0a0a0a")
            plt.close()
            print(f"  ok m87_2017_lo_grid.png (4-day grid)")
    except Exception as e:
        print(f"  FAILED grid: {e}")


def visualize_synthetic_data() -> None:
    """Render synthetic clean/degraded pairs as comparison PNGs."""
    syn_vis_dir = OUTPUT_DIR / "synthetic"
    syn_vis_dir.mkdir(parents=True, exist_ok=True)

    clean_dir = SIMULATED_DIR / "clean"
    degraded_dir = SIMULATED_DIR / "degraded"

    clean_files = sorted(clean_dir.glob("*.npy"))

    if not clean_files:
        print("  No synthetic data found")
        return

    print(f"\n  Synthetic Data Visualization ({len(clean_files)} pairs available)")

    # Pick one sample per degradation level
    levels = {"light": None, "medium": None, "heavy": None, "extreme": None}

    for cf in clean_files:
        for level in levels:
            if level in cf.name and levels[level] is None:
                df = degraded_dir / cf.name
                if df.exists():
                    levels[level] = (cf, df)
                break

    # Side-by-side comparisons
    for level, pair in levels.items():
        if pair is None:
            continue
        cf, df = pair
        clean = np.load(cf)
        degraded = np.load(df)

        save_comparison(
            clean, degraded,
            syn_vis_dir / f"comparison_{level}.png",
            label=f"Degradation: {level.upper()}",
        )
        print(f"  ok comparison_{level}.png")

        save_single_image(clean, syn_vis_dir / f"clean_{level}.png", title=f"Clean — {level}")
        save_single_image(degraded, syn_vis_dir / f"degraded_{level}.png", title=f"Degraded — {level}")

    # All levels in one grid
    grid_pairs = []
    for level in ["light", "medium", "heavy", "extreme"]:
        if levels[level] is not None:
            cf, df = levels[level]
            grid_pairs.append((np.load(cf), np.load(df), level))

    if grid_pairs:
        save_grid(
            grid_pairs,
            syn_vis_dir / "degradation_levels_grid.png",
            title="All Degradation Levels",
        )
        print(f"  ok degradation_levels_grid.png")

    # Crescent vs Ring comparison
    crescent_sample = None
    ring_sample = None
    for cf in clean_files:
        if "crescent_light" in cf.name and crescent_sample is None:
            crescent_sample = np.load(cf)
        if "ring_light" in cf.name and ring_sample is None:
            ring_sample = np.load(cf)
        if crescent_sample is not None and ring_sample is not None:
            break

    if crescent_sample is not None and ring_sample is not None:
        plot_style()
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        axes[0].imshow(crescent_sample, cmap="afmhot", origin="lower", norm=PowerNorm(gamma=0.5))
        axes[0].set_title("Crescent Model", fontsize=13, fontweight="bold")
        axes[0].axis("off")
        axes[1].imshow(ring_sample, cmap="afmhot", origin="lower", norm=PowerNorm(gamma=0.5))
        axes[1].set_title("Ring Model", fontsize=13, fontweight="bold")
        axes[1].axis("off")
        fig.suptitle("Model Types", fontsize=16, fontweight="bold", y=1.02, color="#FF6B35")
        plt.tight_layout(pad=1.0)
        plt.savefig(syn_vis_dir / "model_types.png", dpi=200, bbox_inches="tight", facecolor="#0a0a0a")
        plt.close()
        print(f"  ok model_types.png")


def main() -> None:
    print("=" * 60)
    print("  DeepHorizon - Data Visualization")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    visualize_eht_data()
    visualize_synthetic_data()

    total = len(list(OUTPUT_DIR.rglob("*.png")))
    print(f"\n{'=' * 60}")
    print(f"  Done: {total} images generated")
    print(f"  Dir:  {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
