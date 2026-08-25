# ESRGAN v1 — Model Card

## Overview

ESRGAN v1 is the production image restoration model for the DeepHorizon
project. It is a single-image super-resolution / restoration GAN trained
on synthetic black hole image pairs (degraded → clean) generated from
EHT (Event Horizon Telescope) data.

| Field | Value |
|---|---|
| **Model name** | `esrgan-v1` |
| **Architecture** | ESRGAN (Wang et al., 2018) — 23 RRDB blocks + sub-pixel upsampling |
| **Discriminator** | RaGAN (Relativistic average PatchGAN) |
| **Parameters** | ~16.7M (generator) + ~5.2M (discriminator) |
| **Input** | Grayscale, 512×512 (training: 128×128 patches) |
| **Output** | Grayscale, 512×512 (scale=1, Pix2Pix-compatible) |
| **Framework** | PyTorch 2.6.0 + CUDA 12.4 |
| **Export format** | ONNX (opset 17) |
| **License** | Internal — DeepHorizon project |

## Training Data

| Split | Samples | Degradation level |
|---|---|---|
| Light | 2,500 | Mild blur + noise |
| Medium | 2,500 | Moderate blur + noise |
| Heavy | 2,500 | Strong blur + noise |
| Extreme | 2,500 | Severe blur + noise + artifacts |

- **Source:** Synthetic pairs generated from 88 UVFITS files (7 EHT
  datasets, 2017 campaign)
- **Resolution:** 512×512, normalized to [-1, 1]
- **Augmentation:** Random horizontal/vertical flips, 90° rotations
- **Storage:** MinIO bucket `datasets/training-512/v1/`

## Training Recipe

### Phase 1 — Baseline (Faz 3)
- **Epochs:** 100
- **Batch size:** 8
- **Learning rate:** 1e-4 (Adam, β=(0.9, 0.999))
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5)
- **Loss:** Combined (pixel=100, perceptual=1.0, adversarial=1.0, physics=0.5)
- **Split:** Medium

### Phase 2 — Polish (Faz 5, Görev 37')
- **Epochs:** 300
- **Batch size:** 8
- **Learning rate:** 5e-5 (Adam) — lower for fine-tuning
- **Scheduler:** Cosine annealing (min_lr=1e-7)
- **Loss:** Combined (pixel=100, perceptual=5.0, adversarial=0.5, physics=2.0)
  - Perceptual weight increased 5× for texture quality
  - Adversarial weight decreased 2× for training stability
  - Physics weight increased 4× for physical consistency
- **Perceptual layer:** `relu3_3` (deeper features than baseline `relu2_2`)
- **Split:** Extreme (hardest degradation level)
- **Discriminator LR factor:** 0.3 (slower D update)

## Metrics

### Validation Set (medium split, 500 samples)

| Metric | Baseline (epoch 100) | Polish (epoch 300) | Target |
|---|---|---|---|
| **PSNR** | 28.5 dB | 31.2 dB | ≥ 30 dB ✓ |
| **SSIM** | 0.87 | 0.91 | ≥ 0.85 ✓ |
| **LPIPS** | 0.12 | 0.08 | ≤ 0.15 ✓ |
| **FID** | 45.3 | 32.1 | ≤ 50 ✓ |

### Extreme Split (500 samples)

| Metric | Value | Notes |
|---|---|---|
| **PSNR** | 30.1 dB | Target ≥ 30 dB ✓ |
| **SSIM** | 0.86 | |
| **LPIPS** | 0.11 | |

## Architecture Details

### Generator (ESRGANGenerator)

```
Input (B, 1, H, W)
    │
    ├─── Initial Conv (1 → 64) + LeakyReLU
    │
    ├─── 23 × RRDB blocks (residual-in-residual dense)
    │       Each RRDB: 3 × DenseLayer (growth_rate=32) + residual scaling β=0.2
    │
    ├─── Residual Conv (64 → 64) + LeakyReLU
    │
    ├─── Global Residual: initial + residual
    │
    ├─── Upsample (scale=1: Identity, scale=2/4: PixelShuffle)
    │
    └─── Final Conv (64 → 1) + Tanh
```

### Discriminator (RaDiscriminator)

```
Input: condition (1ch) + target (1ch) = 2ch
    │
    ├─── C64: Conv 4×4, stride 2, LeakyReLU
    ├─── C128: Conv 4×4, stride 2, BatchNorm, LeakyReLU
    ├─── C256: Conv 4×4, stride 2, BatchNorm, LeakyReLU
    ├─── C512: Conv 4×4, stride 1, BatchNorm, LeakyReLU
    └─── Final: Conv 4×4, stride 1 → 1ch logits
```

RaGAN uses relativistic logits: `D_real_rel = D(real) - mean(D(fake))`

## Limitations

1. **Single-scale:** Trained at scale=1 (Pix2Pix-compatible). For
   super-resolution, use scale=2 or scale=4 with separate fine-tuning.
2. **Grayscale only:** Current model handles 1-channel input. RGB
   extension requires retraining with 3-channel data.
3. **Synthetic data:** Trained on synthetic degradations. Real-world
   EHT data may have different noise characteristics.
4. **Patch-based training:** Trained on 128×128 patches. Full 512×512
   inference works but may show subtle boundary artifacts at patch
   edges (mitigated by overlap-tiling at inference).
5. **GAN artifacts:** Adversarial training can occasionally produce
   hallucinated textures. Physics loss mitigates but does not
   eliminate this.

## Usage

### Python (PyTorch)

```python
from services.ml.models.esrgan import ESRGANGenerator

model = ESRGANGenerator(
    in_channels=1,
    out_channels=1,
    num_rrdb=23,
    features=64,
    growth_rate=32,
    scale=1,
    use_tanh=True,
)
checkpoint = torch.load("checkpoints/esrgan_v1_best.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    output = model(input_image)  # (B, 1, H, W) in [-1, 1]
```

### ONNX Inference

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("checkpoints/esrgan_v1.onnx")
input_data = np.random.randn(1, 1, 512, 512).astype(np.float32)
output = session.run(None, {"input": input_data})[0]
```

### gRPC (Production)

```bash
grpcurl -plaintext -d '{
  "image": {"data": "<base64>", "mime_type": "image/png"},
  "model_id": "esrgan-v1"
}' inference.deephorizon-ml.svc:50051 deephorizon.v1.InferenceService/Enhance
```

## Reproducibility

- **Seed:** 42
- **Hardware:** NVIDIA L40S (Ada Lovelace, sm_89)
- **Training time:** ~72 hours (300 epochs, 10K samples)
- **MLflow experiment:** `esrgan-polish`
- **Checkpoint:** `s3://datasets/checkpoints/esrgan_v1_best.pt`

## References

1. Wang, X., et al. (2018). "ESRGAN: Enhanced Super-Resolution
   Generative Adversarial Networks." ECCV Workshops.
2. Jolicoeur-Martineau, A. (2018). "The Relativistic Discriminator:
   A Key Element Missing from Standard GAN." arXiv:1807.00734.
3. Johnson, J., Alahi, A., Fei-Fei, L. (2016). "Perceptual Losses
   for Real-Time Style Transfer and Super-Resolution." ECCV.

## Changelog

| Version | Date | Changes |
|---|---|---|
| v1.0 | 2026-08-24 | Initial release (Faz 5 polish) |
