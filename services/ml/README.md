# services/ml

Python package for model training, evaluation, and gRPC inference serving.

**Owners:** Stajyer 3 (baseline + GAN), Stajyer 4 (SOTA + physics loss), Stajyer 5 (evaluation + inference).

## Layout

| Path | Purpose |
|:---|:---|
| `models/` | Network architectures — `unet/`, `pix2pix/`, `esrgan/`, `restormer/` |
| `losses/` | `physics.py`, `perceptual.py`, `gan.py` |
| `data/` | Datasets, dataloaders, transforms |
| `training/` | `train_loop.py`, `optuna_runner.py`, Hydra configs |
| `evaluation/` | `metrics.py` (PSNR/SSIM/LPIPS/FID + physics), `benchmark.py` |
| `inference_server/` | gRPC server implementation (consumes `proto/`) |

## Install

```bash
uv sync --extra ml --extra dev
```

> Do not mix with the `data` extra in the same venv — see root README.
