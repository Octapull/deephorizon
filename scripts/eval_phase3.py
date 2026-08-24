"""
DeepHorizon — Faz 3 Değerlendirme Scripti
==========================================
ESRGAN (veya Pix2Pix) modelini medium split üzerinde değerlendirir ve
SSIM ≥ 0.85 go/no-go kararı verir.

README'deki Faz 3 doğrulama kriteri:
    "ESRGAN 200 epoch sonra medium split'te SSIM ≥ 0.85"

Kullanım:
    # ESRGAN checkpoint ile
    python scripts/eval_phase3.py \\
        --checkpoint checkpoints/esrgan_best.pt \\
        --architecture esrgan \\
        --split medium \\
        --output-json outputs/phase3_eval.json

    # Pix2Pix checkpoint ile
    python scripts/eval_phase3.py \\
        --checkpoint checkpoints/pix2pix_best.pt \\
        --architecture pix2pix \\
        --split medium

Çıktı:
    - Konsol tablosu (PSNR/SSIM/LPIPS/FID + physics metrikleri)
    - JSON dosyası (--output-json ile belirtilirse)
    - ADR taslağı (docs/adr/0002-phase4-decision.md)

DRY: Metrik hesaplama `services.ml.evaluation.metrics` modülünden,
    model yükleme `services.ml.export.onnx_export` modülünden.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Proje kökünü path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.ml.data.dataset import BlackHoleDataset
from services.ml.evaluation.metrics import (
    compute_fid,
    compute_metrics,
    compute_physics_metrics,
)

# ---------------------------------------------------------------------------
# Faz 3 go/no-go gate
# ---------------------------------------------------------------------------
SSIM_GATE = 0.85  # README'deki Faz 3 doğrulama kriteri
PSNR_TARGET = 28.0  # Bilgilendirme amaçlı (zorunlu gate değil)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Faz 3 model değerlendirmesi + go/no-go kararı",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Model checkpoint yolu (.pt)",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["unet", "pix2pix", "esrgan"],
        default="esrgan",
        help="Model mimarisi (default: esrgan)",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="data/training",
        help="Yerel veri kök dizini",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["light", "medium", "heavy", "extreme"],
        default="medium",
        help="Degradation split (default: medium)",
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
        default=8,
        help="Batch boyutu (default: 8 — ESRGAN bellek yoğun)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Sonuçları JSON olarak kaydet",
    )
    parser.add_argument(
        "--adr-output",
        type=str,
        default="docs/adr/0002-phase4-decision.md",
        help="ADR çıktı yolu (go/no-go kararı)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def load_model(
    architecture: str, checkpoint_path: Path, device: torch.device
) -> torch.nn.Module:
    """Mimariye göre modeli yükle ve eval moduna al.

    Mimariye özel state_dict key'leri farklı olabilir; bu yüzden
    her mimari için ayrı yükleme yapılır.
    """
    if architecture == "unet":
        from services.ml.models.unet import UNet

        model = UNet(in_channels=1, out_channels=1)
    elif architecture == "pix2pix":
        from services.ml.models.pix2pix import Pix2PixGenerator

        model = Pix2PixGenerator(in_channels=1, out_channels=1)
    elif architecture == "esrgan":
        from services.ml.models.esrgan import ESRGANGenerator

        model = ESRGANGenerator(in_channels=1, out_channels=1, scale_factor=1)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Eğer checkpoint {"generator_state_dict": ...} formatındaysa aç
    if isinstance(state_dict, dict) and "generator_state_dict" in state_dict:
        state_dict = state_dict["generator_state_dict"]
    elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_phase3(args: argparse.Namespace) -> dict:
    """Ana değerlendirme döngüsü.

    Returns:
        Dict with metrics + go/no-go kararı.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadı: {checkpoint_path}")

    print(f"[eval_phase3] Architecture: {args.architecture}")
    print(f"[eval_phase3] Checkpoint: {checkpoint_path}")
    print(f"[eval_phase3] Device: {device}")
    print(f"[eval_phase3] Split: {args.split}")

    # Model yükle
    model = load_model(args.architecture, checkpoint_path, device)
    print(
        f"[eval_phase3] Model yüklendi: {sum(p.numel() for p in model.parameters()):,} parametre"
    )

    # Dataset
    dataset = BlackHoleDataset(
        root_dir=args.root_dir,
        use_minio=False,
        augment=False,
        split=args.split,
    )

    if len(dataset) == 0:
        raise FileNotFoundError(
            f"Dataset boş: {args.root_dir}/{args.split}/clean/*.npy bulunamadı. "
            f"Önce `python scripts/generate_training_data.py` çalıştırın."
        )

    num_samples = (
        min(args.num_samples, len(dataset)) if args.num_samples else len(dataset)
    )
    print(
        f"[eval_phase3] Toplam örnek: {len(dataset)}, değerlendirilecek: {num_samples}"
    )

    # Accumulators
    psnr_sum = ssim_sum = lpips_sum = 0.0
    flux_sum = ring_sum = asym_sum = 0.0
    count = 0
    real_features_list = []
    fake_features_list = []

    start = time.time()

    for idx in range(num_samples):
        degraded, clean = dataset[idx]
        degraded = degraded.unsqueeze(0).to(device)
        clean = clean.unsqueeze(0).to(device)

        # Model inference
        prediction = model(degraded)

        # Boyut uyumsuzluğu varsa clean'i prediction boyutuna getir
        if prediction.shape != clean.shape:
            clean = F.interpolate(
                clean, size=prediction.shape[-2:], mode="bilinear", align_corners=False
            )

        # Clamp [0, 1] — model çıktısı bazen dışarı taşabilir
        prediction = prediction.clamp(0.0, 1.0)

        # Metrikler
        m = compute_metrics(
            prediction,
            clean,
            data_range=1.0,
            include_lpips=True,
            include_physics=True,
        )
        psnr_sum += m["psnr"]
        ssim_sum += m["ssim"]
        lpips_sum += m["lpips"]
        flux_sum += m["flux_error"]
        ring_sum += m["ring_diameter_error"]
        asym_sum += m["asymmetry_error"]

        real_features_list.append(clean)
        fake_features_list.append(prediction)

        count += 1
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f"  [{idx + 1}/{num_samples}] elapsed: {elapsed:.1f}s")

    # Ortalamalar
    psnr_avg = psnr_sum / count
    ssim_avg = ssim_sum / count
    lpips_avg = lpips_sum / count
    flux_avg = flux_sum / count
    ring_avg = ring_sum / count
    asym_avg = asym_sum / count

    # FID
    print("[eval_phase3] FID hesaplanıyor...")
    real_all = torch.cat(real_features_list, dim=0)
    fake_all = torch.cat(fake_features_list, dim=0)
    fid_value = compute_fid(real_all, fake_all)

    elapsed = time.time() - start

    # Go/no-go kararı
    decision = "GO" if ssim_avg >= SSIM_GATE else "NO-GO"
    margin = ssim_avg - SSIM_GATE

    results = {
        "architecture": args.architecture,
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "num_samples": count,
        "psnr": psnr_avg,
        "ssim": ssim_avg,
        "lpips": lpips_avg,
        "fid": fid_value,
        "flux_error": flux_avg,
        "ring_diameter_error": ring_avg,
        "asymmetry_error": asym_avg,
        "ssim_gate": SSIM_GATE,
        "ssim_margin": margin,
        "decision": decision,
        "elapsed_sec": elapsed,
        "device": str(device),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return results


def print_table(results: dict) -> None:
    """Konsol tablosu yazdır + go/no-go kararını göster."""
    print("\n" + "=" * 70)
    print(
        f"FAZ 3 DEĞERLENDİRME — {results['architecture'].upper()} ({results['split'].upper()})"
    )
    print("=" * 70)
    print(f"{'Metric':<22} {'Value':>12} {'Gate':>12} {'Status':>10}")
    print("-" * 70)

    rows = [
        ("PSNR (dB)", results["psnr"], PSNR_TARGET, "INFO"),
        ("SSIM", results["ssim"], SSIM_GATE, "GATE"),
        ("LPIPS", results["lpips"], None, "INFO"),
        ("FID", results["fid"], None, "INFO"),
        ("Flux error", results["flux_error"], None, "INFO"),
        ("Ring diameter err", results["ring_diameter_error"], None, "INFO"),
        ("Asymmetry err", results["asymmetry_error"], None, "INFO"),
    ]
    for name, value, gate, kind in rows:
        if gate is not None:
            status = "✅ PASS" if value >= gate else "❌ FAIL"
            gate_str = f">= {gate:.2f}"
        else:
            status = "—"
            gate_str = "—"
        print(f"{name:<22} {value:>12.4f} {gate_str:>12} {status:>10}")

    print("-" * 70)
    decision = results["decision"]
    margin = results["ssim_margin"]
    print(f"SSIM margin: {margin:+.4f} (gate: {SSIM_GATE})")
    print(f"Decision:    {'🟢 GO' if decision == 'GO' else '🔴 NO-GO'}")
    print(f"Elapsed:     {results['elapsed_sec']:.1f}s | Device: {results['device']}")
    print("=" * 70)


def write_adr(results: dict, adr_path: Path) -> None:
    """Go/no-go kararına göre ADR dosyası yaz."""
    decision = results["decision"]
    decision_emoji = "🟢" if decision == "GO" else "🔴"

    adr_content = f"""# ADR 0002 — Faz 4 Go/No-Go Kararı

## Bağlam
Faz 3 tamamlandı: ESRGAN modeli eğitildi ve medium split üzerinde
değerlendirildi. Faz 4'e (Go API + Frontend entegrasyonu) geçiş için
SSIM ≥ {SSIM_GATE} gate'i konulmuştu.

## Değerlendirme Sonuçları

| Metric | Value | Gate | Status |
|--------|------:|-----:|:------:|
| **SSIM** | **{results['ssim']:.4f}** | >= {SSIM_GATE} | {'✅ PASS' if results['ssim'] >= SSIM_GATE else '❌ FAIL'} |
| PSNR | {results['psnr']:.4f} dB | >= {PSNR_TARGET} | {'✅' if results['psnr'] >= PSNR_TARGET else '⚠️'} |
| LPIPS | {results['lpips']:.4f} | — | — |
| FID | {results['fid']:.4f} | — | — |
| Flux error | {results['flux_error']:.4f} | — | — |
| Ring diameter error | {results['ring_diameter_error']:.4f} | — | — |
| Asymmetry error | {results['asymmetry_error']:.4f} | — | — |

**Değerlendirme detayları:**
- Mimari: `{results['architecture']}`
- Checkpoint: `{results['checkpoint']}`
- Split: `{results['split']}` ({results['num_samples']} örnek)
- Cihaz: `{results['device']}`
- Süre: {results['elapsed_sec']:.1f}s
- Zaman damgası: {results['timestamp']}

## Karar

{decision_emoji} **{decision}** — SSIM = {results['ssim']:.4f} (gate: {SSIM_GATE}, margin: {results['ssim_margin']:+.4f})

"""

    if decision == "GO":
        adr_content += """## Gerekçe
SSIM gate'i karşılandı. Faz 4'e (Go API + Frontend entegrasyonu) geçiş
yapılabilir.

## Sonuçlar
- Go API ekibi `services/api/internal/jobs/` async job queue'yu aktive edebilir
- Frontend ekibi `services/frontend/` MVP'yi tamamlayabilir
- Inference server (`services/ml/inference_server/`) production'a alınabilir
- Monitoring (`infra/k8s/monitor/`) aktifleştirilebilir

## Aksiyonlar
- [ ] Go API: `POST /enhance` → gRPC Enhance entegrasyonu (Görev 29)
- [ ] Go API: client pool + retry/backoff (Görev 30)
- [ ] Go API: async job queue (Görev 31)
- [ ] Frontend: Next.js 15 MVP (Görev 33)
- [ ] CI/CD: GitHub Actions pipeline aktifleştir
"""
    else:
        adr_content += f"""## Gerekçe
SSIM gate'i ({SSIM_GATE}) karşılanamadı. Mevcut SSIM = {results['ssim']:.4f}
(margin: {results['ssim_margin']:+.4f}). Faz 4'e geçmeden önce kök neden
analizi yapılmalı.

## Olası Kök Nedenler
1. **Yetersiz eğitim süresi**: 200 epoch hedefine ulaşılamamış olabilir
2. **Hiperparametre optimizasyonu**: Optuna runner ile yeniden arama gerekli
3. **Veri kalitesi**: medium split'te bozulma seviyesi çok yüksek olabilir
4. **Model kapasitesi**: ESRGAN yerine daha büyük mimari gerekebilir

## Aksiyonlar
- [ ] Eğitim loglarını MLflow'dan incele (overfit/underfit analizi)
- [ ] Optuna runner ile 50+ trial çalıştır
- [ ] Learning rate ve loss weight'leri yeniden ayarla
- [ ] Gerekirse Pix2Pix fallback'i ile Faz 4'e geç
- [ ] Bu ADR'yi güncelle ve yeniden değerlendir
"""

    adr_path.parent.mkdir(parents=True, exist_ok=True)
    adr_path.write_text(adr_content, encoding="utf-8")
    print(f"\n[eval_phase3] ADR yazıldı: {adr_path}")


def main() -> None:
    args = parse_args()
    results = evaluate_phase3(args)
    print_table(results)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[eval_phase3] Sonuçlar kaydedildi: {output_path}")

    # ADR yaz
    adr_path = Path(args.adr_output)
    write_adr(results, adr_path)

    # Exit code: GO ise 0, NO-GO ise 1 (CI/CD pipeline'lar için)
    sys.exit(0 if results["decision"] == "GO" else 1)


if __name__ == "__main__":
    main()
