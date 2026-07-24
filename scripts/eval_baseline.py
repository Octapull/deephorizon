"""
DeepHorizon — Baseline Evaluation Script
=========================================
Bicubic upsample baseline + tüm metrikler (PSNR, SSIM, LPIPS, FID).

README'deki baseline rakamlarını üretmek için kullanılır:
- PSNR ~18 dB, SSIM ~0.35, LPIPS ~0.55, FID ~180 (medium split, 2500 pairs)

Kullanım:
    python scripts/eval_baseline.py --split medium --num-samples 2500 \\
        --output-json baseline_medium.json

DRY: Bu script `services.ml.evaluation.metrics` modülünü kullanır — metrik
hesaplama mantığı burada yeniden yazılmaz.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Proje kökünü path'e ekle (script doğrudan çalıştırılabilsin)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.ml.data.dataset import BlackHoleDataset
from services.ml.evaluation.metrics import (
    compute_fid,
    compute_lpips,
    compute_metrics,
    compute_psnr,
    compute_ssim,
)


# README'deki baseline rakamları (karşılaştırma için)
EXPECTED_BASELINE = {
    "psnr": 18.0,
    "ssim": 0.35,
    "lpips": 0.55,
    "fid": 180.0,
}

# Toleranslar (±)
TOLERANCE = {
    "psnr": 0.5,
    "ssim": 0.05,
    "lpips": 0.05,
    "fid": 20.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bicubic upsample baseline + PSNR/SSIM/LPIPS/FID evaluation",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="data/training",
        help="Yerel veri kök dizini (default: data/training)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["light", "medium", "heavy", "extreme"],
        default="medium",
        help="Degradation split (default: medium)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Sonuçları JSON olarak kaydet (örn: baseline_medium.json)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Değerlendirilecek örnek sayısı (None = tüm split)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch boyutu (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Cihaz (default: cuda varsa cuda)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def bicubic_upsample(degraded: torch.Tensor, scale_factor: int = 2) -> torch.Tensor:
    """Bicubic upsample ile degraded görüntüyü hedef boyuta getir.

    Args:
        degraded: (B, C, H, W) tensor.
        scale_factor: Upsample çarpanı (default: 2).

    Returns:
        (B, C, H*scale_factor, W*scale_factor) tensor.
    """
    return F.interpolate(degraded, scale_factor=scale_factor, mode="bicubic", align_corners=False)


def evaluate_baseline(args: argparse.Namespace) -> dict:
    """Ana değerlendirme döngüsü.

    Args:
        args: argparse namespace.

    Returns:
        Dict with keys: psnr, ssim, lpips, fid, num_samples, elapsed_sec.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"[eval_baseline] Device: {device}")
    print(f"[eval_baseline] Split: {args.split}")
    print(f"[eval_baseline] Root dir: {args.root_dir}")

    # Dataset — augmentation KAPALI, sadece split
    dataset = BlackHoleDataset(
        root_dir=args.root_dir,
        use_minio=False,
        augment=False,
        split=args.split,
    )

    if len(dataset) == 0:
        raise FileNotFoundError(
            f"Dataset bos: {args.root_dir}/{args.split}/clean/*.npy bulunamadi. "
            f"Once `python scripts/generate_training_data.py` calistirin."
        )

    num_samples = min(args.num_samples, len(dataset)) if args.num_samples else len(dataset)
    print(f"[eval_baseline] Toplam ornek: {len(dataset)}, degerlendirilecek: {num_samples}")

    # Accumulators
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    count = 0

    # FID için tüm görüntüleri topla (InceptionV3 pool_features)
    real_features_list = []
    fake_features_list = []

    start = time.time()

    for idx in range(num_samples):
        degraded, clean = dataset[idx]
        # (1, 1, H, W) — batch dimension ekle
        degraded = degraded.unsqueeze(0).to(device)
        clean = clean.unsqueeze(0).to(device)

        # Bicubic upsample
        upsampled = bicubic_upsample(degraded, scale_factor=2)

        # Upsampled ile clean aynı boyutta olmalı
        if upsampled.shape != clean.shape:
            # Boyut uyumsuzsa clean'i upsampled boyutuna getir
            clean = F.interpolate(clean, size=upsampled.shape[-2:], mode="bilinear", align_corners=False)

        # Metrikler
        m = compute_metrics(
            upsampled, clean,
            data_range=1.0,
            include_lpips=True,
            include_physics=False,
        )
        psnr_sum += m["psnr"]
        ssim_sum += m["ssim"]
        lpips_sum += m["lpips"]

        # FID için Inception features topla
        real_features_list.append(clean)
        fake_features_list.append(upsampled)

        count += 1
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f"  [{idx + 1}/{num_samples}] elapsed: {elapsed:.1f}s")

    # Ortalamalar
    psnr_avg = psnr_sum / count
    ssim_avg = ssim_sum / count
    lpips_avg = lpips_sum / count

    # FID — tüm görüntüleri birleştir
    print("[eval_baseline] FID hesaplaniyor...")
    real_all = torch.cat(real_features_list, dim=0)
    fake_all = torch.cat(fake_features_list, dim=0)
    fid_value = compute_fid(real_all, fake_all)

    elapsed = time.time() - start

    results = {
        "split": args.split,
        "num_samples": count,
        "psnr": psnr_avg,
        "ssim": ssim_avg,
        "lpips": lpips_avg,
        "fid": fid_value,
        "elapsed_sec": elapsed,
        "device": str(device),
    }

    return results


def print_table(results: dict) -> None:
    """Konsol tablosu yazdir + README baseline ile karsilastir."""
    print("\n" + "=" * 70)
    print(f"BICUBIC BASELINE — {results['split'].upper()} SPLIT ({results['num_samples']} samples)")
    print("=" * 70)
    print(f"{'Metric':<10} {'Measured':>12} {'README':>12} {'Delta':>10} {'Status':>10}")
    print("-" * 70)

    for metric in ["psnr", "ssim", "lpips", "fid"]:
        measured = results[metric]
        expected = EXPECTED_BASELINE[metric]
        tol = TOLERANCE[metric]
        delta = abs(measured - expected)
        status = "OK" if delta <= tol else "WARN"
        print(f"{metric.upper():<10} {measured:>12.4f} {expected:>12.4f} {delta:>10.4f} {status:>10}")

    print("-" * 70)
    print(f"Elapsed: {results['elapsed_sec']:.1f}s | Device: {results['device']}")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    results = evaluate_baseline(args)
    print_table(results)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[eval_baseline] Sonuclar kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
