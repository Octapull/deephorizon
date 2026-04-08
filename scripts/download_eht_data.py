"""
DeepHorizon - EHT Data Downloader
==================================
Downloads all publicly released calibrated UVFITS visibility data
from the Event Horizon Telescope collaboration.

Sources:
  - M87* 2017 (Paper I-VI)
  - 3C279 2017
  - Sgr A* 2017 (Paper I-VIII)
  - M87* 2018
  - Cen A 2017
  - M87* Polarized 2017
  - Sgr A* Polarized 2017
"""

import argparse
import requests
from pathlib import Path
from typing import Dict


DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "eht"


def _gh(repo: str, branch: str, path: str) -> str:
    return f"https://github.com/eventhorizontelescope/{repo}/raw/{branch}/{path}"


# ──────────────────────────────────────────────────────────────
# All publicly available EHT datasets
# ──────────────────────────────────────────────────────────────

EHT_DATASETS: Dict[str, dict] = {
    # ── M87* 2017 ─────────────────────────────────────────────
    "m87_2017": {
        "description": "M87* — 2017 EHT (Paper IV, 8 files)",
        "repo": "2019-D01-01",
        "branch": "master",
        "files": [
            "uvfits/SR1_M87_2017_095_lo_hops_netcal_StokesI.uvfits",
            "uvfits/SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits",
            "uvfits/SR1_M87_2017_096_lo_hops_netcal_StokesI.uvfits",
            "uvfits/SR1_M87_2017_096_hi_hops_netcal_StokesI.uvfits",
            "uvfits/SR1_M87_2017_100_lo_hops_netcal_StokesI.uvfits",
            "uvfits/SR1_M87_2017_100_hi_hops_netcal_StokesI.uvfits",
            "uvfits/SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits",
            "uvfits/SR1_M87_2017_101_hi_hops_netcal_StokesI.uvfits",
        ],
    },

    # ── 3C279 2017 ────────────────────────────────────────────
    "3c279_2017": {
        "description": "3C279 — 2017 EHT (8 files)",
        "repo": "2020-D01-01",
        "branch": "master",
        "files": [
            "uvfits/SR1_3C279_2017_095_lo_hops_netcal_StokesI.uvfits",
            "uvfits/SR1_3C279_2017_095_hi_hops_netcal_StokesI.uvfits",
            "uvfits/SR1_3C279_2017_096_lo_hops_netcal_StokesI.uvfits",
            "uvfits/SR1_3C279_2017_096_hi_hops_netcal_StokesI.uvfits",
            "uvfits/SR1_3C279_2017_100_lo_hops_netcal_StokesI.uvfits",
            "uvfits/SR1_3C279_2017_100_hi_hops_netcal_StokesI.uvfits",
            "uvfits/SR1_3C279_2017_101_lo_hops_netcal_StokesI.uvfits",
            "uvfits/SR1_3C279_2017_101_hi_hops_netcal_StokesI.uvfits",
        ],
    },

    # ── Sgr A* 2017 (Stokes I) ───────────────────────────────
    "sgra_2017": {
        "description": "Sgr A* — 2017 EHT Stokes I (20 files, CASA + HOPS)",
        "repo": "2022-D02-01",
        "branch": "main",
        "files": [
            "uvfits/ER6_SGRA_2017_096_lo_hops_netcal-LMTcal_StokesI.uvfits",
            "uvfits/ER6_SGRA_2017_096_hi_hops_netcal-LMTcal_StokesI.uvfits",
            "uvfits/ER6_SGRA_2017_096_lo_casa_netcal-LMTcal_StokesI.uvfits",
            "uvfits/ER6_SGRA_2017_096_hi_casa_netcal-LMTcal_StokesI.uvfits",
            "uvfits/ER6_SGRA_2017_097_lo_hops_netcal-LMTcal_StokesI.uvfits",
            "uvfits/ER6_SGRA_2017_097_hi_hops_netcal-LMTcal_StokesI.uvfits",
            "uvfits/ER6_SGRA_2017_097_lo_casa_netcal-LMTcal_StokesI.uvfits",
            "uvfits/ER6_SGRA_2017_097_hi_casa_netcal-LMTcal_StokesI.uvfits",
            "uvfits/ER6_SGRA_2017_099_lo_hops_netcal-LMTcal_StokesI.uvfits",
            "uvfits/ER6_SGRA_2017_099_hi_hops_netcal-LMTcal_StokesI.uvfits",
            "uvfits/ER6_SGRA_2017_099_lo_casa_netcal-LMTcal_StokesI.uvfits",
            "uvfits/ER6_SGRA_2017_099_hi_casa_netcal-LMTcal_StokesI.uvfits",
            "uvfits/ER6_SGRA_2017_100_lo_hops_netcal-LMTcal_StokesI.uvfits",
            "uvfits/ER6_SGRA_2017_100_hi_hops_netcal-LMTcal_StokesI.uvfits",
            "uvfits/ER6_SGRA_2017_100_lo_casa_netcal-LMTcal_StokesI.uvfits",
            "uvfits/ER6_SGRA_2017_100_hi_casa_netcal-LMTcal_StokesI.uvfits",
            "uvfits/ER6_SGRA_2017_101_lo_hops_netcal-LMTcal_StokesI.uvfits",
            "uvfits/ER6_SGRA_2017_101_hi_hops_netcal-LMTcal_StokesI.uvfits",
            "uvfits/ER6_SGRA_2017_101_lo_casa_netcal-LMTcal_StokesI.uvfits",
            "uvfits/ER6_SGRA_2017_101_hi_casa_netcal-LMTcal_StokesI.uvfits",
        ],
    },

    # ── M87* 2018 ─────────────────────────────────────────────
    "m87_2018": {
        "description": "M87* — 2018 EHT (24 files, 4 bands x CASA + HOPS)",
        "repo": "2024-D01-01",
        "branch": "main",
        "files": [
            "uvfits/L2V1_M87_2018_111_b1_hops_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_111_b1_casa_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_111_b2_hops_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_111_b2_casa_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_111_b3_hops_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_111_b3_casa_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_111_b4_hops_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_111_b4_casa_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_112_b1_hops_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_112_b1_casa_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_112_b2_hops_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_112_b2_casa_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_112_b3_hops_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_112_b3_casa_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_112_b4_hops_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_112_b4_casa_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_113_b1_hops_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_113_b1_casa_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_113_b2_hops_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_113_b2_casa_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_113_b3_hops_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_113_b3_casa_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_113_b4_hops_netcal_10s_StokesI.uvfits",
            "uvfits/L2V1_M87_2018_113_b4_casa_netcal_10s_StokesI.uvfits",
        ],
    },

    # ── Centaurus A 2017 ──────────────────────────────────────
    "cena_2017": {
        "description": "Centaurus A — 2017 EHT (4 files)",
        "repo": "2021-D03-01",
        "branch": "main",
        "files": [
            "uvfits/CenA_2017_100_lo_hops_netcal_StokesI.uvfits",
            "uvfits/CenA_2017_100_hi_hops_netcal_StokesI.uvfits",
            "uvfits/CenA_2017_100_lo_casa_netcal_StokesI.uvfits",
            "uvfits/CenA_2017_100_hi_casa_netcal_StokesI.uvfits",
        ],
    },

    # ── M87* 2017 Polarized ───────────────────────────────────
    "m87_2017_pol": {
        "description": "M87* — 2017 Polarized (16 files, CASA + HOPS)",
        "repo": "2023-D01-01",
        "branch": "main",
        "files": [
            "casa_data/April05/SR2_M87_2017_095_hi_casa_EVPA_rotation+dcal+ampscale+netcal+10s-avg+RLgain-amp.uvfits",
            "casa_data/April05/SR2_M87_2017_095_lo_casa_EVPA_rotation+dcal+ampscale+netcal+10s-avg+RLgain-amp.uvfits",
            "casa_data/April06/SR2_M87_2017_096_hi_casa_EVPA_rotation+dcal+ampscale+netcal+10s-avg+RLgain-amp.uvfits",
            "casa_data/April06/SR2_M87_2017_096_lo_casa_EVPA_rotation+dcal+ampscale+netcal+10s-avg+RLgain-amp.uvfits",
            "casa_data/April10/SR2_M87_2017_100_hi_casa_EVPA_rotation+dcal+ampscale+netcal+10s-avg+RLgain-amp.uvfits",
            "casa_data/April10/SR2_M87_2017_100_lo_casa_EVPA_rotation+dcal+ampscale+netcal+10s-avg+RLgain-amp.uvfits",
            "casa_data/April11/SR2_M87_2017_101_hi_casa_EVPA_rotation+dcal+ampscale+netcal+10s-avg+RLgain-amp.uvfits",
            "casa_data/April11/SR2_M87_2017_101_lo_casa_EVPA_rotation+dcal+ampscale+netcal+10s-avg+RLgain-amp.uvfits",
            "hops_data/April05/SR2_M87_2017_095_hi_hops_EVPA_rotation+dcal+ampscale+netcal+10s-avg+RLgain-amp.uvfits",
            "hops_data/April05/SR2_M87_2017_095_lo_hops_EVPA_rotation+dcal+ampscale+netcal+10s-avg+RLgain-amp.uvfits",
            "hops_data/April06/SR2_M87_2017_096_hi_hops_EVPA_rotation+dcal+ampscale+netcal+10s-avg+RLgain-amp.uvfits",
            "hops_data/April06/SR2_M87_2017_096_lo_hops_EVPA_rotation+dcal+ampscale+netcal+10s-avg+RLgain-amp.uvfits",
            "hops_data/April10/SR2_M87_2017_100_hi_hops_EVPA_rotation+dcal+ampscale+netcal+10s-avg+RLgain-amp.uvfits",
            "hops_data/April10/SR2_M87_2017_100_lo_hops_EVPA_rotation+dcal+ampscale+netcal+10s-avg+RLgain-amp.uvfits",
            "hops_data/April11/SR2_M87_2017_101_hi_hops_EVPA_rotation+dcal+ampscale+netcal+10s-avg+RLgain-amp.uvfits",
            "hops_data/April11/SR2_M87_2017_101_lo_hops_EVPA_rotation+dcal+ampscale+netcal+10s-avg+RLgain-amp.uvfits",
        ],
    },

    # ── Sgr A* 2017 Polarized ─────────────────────────────────
    "sgra_2017_pol": {
        "description": "Sgr A* — 2017 Polarized (8 files, CASA + HOPS)",
        "repo": "2024-D02-01",
        "branch": "main",
        "files": [
            "casa_data/April06/ER6_SGRA_2017_096_lo_casa_netcal_LMTcal_10s_ALMArot_dtermcal.uvfits",
            "casa_data/April06/ER6_SGRA_2017_096_hi_casa_netcal_LMTcal_10s_ALMArot_dtermcal.uvfits",
            "casa_data/April07/ER6_SGRA_2017_097_lo_casa_netcal_LMTcal_10s_ALMArot_dtermcal.uvfits",
            "casa_data/April07/ER6_SGRA_2017_097_hi_casa_netcal_LMTcal_10s_ALMArot_dtermcal.uvfits",
            "hops_data/April06/ER6_SGRA_2017_096_lo_hops_netcal_LMTcal_10s_ALMArot_dtermcal.uvfits",
            "hops_data/April06/ER6_SGRA_2017_096_hi_hops_netcal_LMTcal_10s_ALMArot_dtermcal.uvfits",
            "hops_data/April07/ER6_SGRA_2017_097_lo_hops_netcal_LMTcal_10s_ALMArot_dtermcal.uvfits",
            "hops_data/April07/ER6_SGRA_2017_097_hi_hops_netcal_LMTcal_10s_ALMArot_dtermcal.uvfits",
        ],
    },
}


def download_file(url: str, dest: Path) -> bool:
    """Download a single file. Skips if already exists."""
    if dest.exists():
        print(f"  [SKIP] {dest.name} (already exists)")
        return True

    print(f"  [DOWN] {dest.name} ...")
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"         ok {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"         FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="EHT UVFITS data downloader")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=f"Datasets to download (empty = all). Options: {', '.join(EHT_DATASETS.keys())}",
    )
    args = parser.parse_args()

    selected = args.datasets if args.datasets else list(EHT_DATASETS.keys())

    total_files = sum(len(EHT_DATASETS[k]["files"]) for k in selected if k in EHT_DATASETS)

    print("=" * 60)
    print("  DeepHorizon - EHT Data Download")
    print(f"  {len(selected)} datasets, {total_files} files")
    print("=" * 60)

    total, success, skipped, failed = 0, 0, 0, 0

    for dataset_key in selected:
        if dataset_key not in EHT_DATASETS:
            print(f"\n  Unknown dataset: {dataset_key}")
            continue

        ds = EHT_DATASETS[dataset_key]
        dataset_dir = DATA_DIR / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  {ds['description']}")
        print(f"   Dir: {dataset_dir}")

        for file_path in ds["files"]:
            total += 1
            filename = Path(file_path).name
            url = _gh(ds["repo"], ds["branch"], file_path)
            dest = dataset_dir / filename

            if dest.exists():
                skipped += 1
                print(f"  [SKIP] {filename}")
            elif download_file(url, dest):
                success += 1
            else:
                failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Total:    {total}")
    print(f"  Downloaded: {success}")
    print(f"  Skipped:  {skipped}")
    print(f"  Failed:   {failed}")
    print(f"  Dir:      {DATA_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
