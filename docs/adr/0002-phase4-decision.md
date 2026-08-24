# ADR 0005 — Phase 4 Go/No-Go Gate Decision: Go (Restormer Added)

## Context

The project plan defined a **Go/No-Go gate** at the end of Phase 4
(weeks 7-8). The gate was designed to answer the following questions:

1. Does the Go API → Python gRPC pipeline work end-to-end?
2. Is the Frontend MVP ready?
3. Did ESRGAN achieve SSIM ≥ 0.85 on the medium split?
4. Are time and resources sufficient?

In the **Go** scenario, Phase 5 would add Restormer (a Transformer-based
image restoration model). In the **No-Go** scenario, the scope would be
narrowed to ESRGAN polish only.

A code-base inspection revealed that Phase 4 infrastructure was largely
complete:

- `services/api/internal/grpc_client/` — Go gRPC client present
- `services/api/internal/jobstore/` — Redis job store present
- `services/api/internal/metrics/` — Prometheus metrics present
- `services/ml/models/esrgan/` — ESRGAN model (rrdb, generator, discriminator) present
- `services/ml/models/pix2pix/` — Pix2Pix model present
- `services/ml/losses/` — GAN, perceptual, physics loss packages present
- `services/ml/training/optuna_runner.py` — Optuna hyperparameter search present

## Decision

**The gate was revised to Go.** The Restormer model was added in Phase 5:

- `services/ml/models/restormer/mdta.py` — Multi-Dconv Head Transposed
  Self-Attention (channel-wise attention, O(C²) cost)
- `services/ml/models/restormer/gdfn.py` — Gated-Dconv Feed-forward
  Network (gating + local context)
- `services/ml/models/restormer/restormer.py` — 4-level encoder-decoder
  transformer (54.3M parameters)
- `services/ml/models/restormer/__init__.py` — `build_restormer()` factory

ESRGAN polish work will continue **in parallel** (both models will be
available in the inference server).

## Rationale

- **Infrastructure ready**: Phase 4's Go API, gRPC, Redis job store, and
  Prometheus metrics are complete, so there is no technical blocker for
  moving to Phase 5's Restormer target.
- **Compute resources suitable**: An L40S GPU is available; Restormer
  training (in 128×128 patches) can be completed in a reasonable time.
- **Research value**: Restormer is a state-of-the-art model that applies
  Transformer-based architecture to image restoration (CVPR 2022). It
  strengthens the research dimension of the project.
- **Risk management**: ESRGAN polish continues in parallel, so even if
  Restormer fails, a production-ready model (ESRGAN) remains available.

## Consequences

- The `services/ml/models/restormer/` package was added (4 files, ~400 lines).
- The model was verified at 128×128 resolution (forward + backward pass).
- 512×512 resolution will be handled during training via a patch-based
  approach (e.g., 128×128 patches with sliding window) — the approach
  recommended in the Restormer paper.
- The remaining Phase 5 tasks (40-41) will be updated according to this
  decision:
  - **Task 40**: `train_restormer.py` — patch-based training, full loss
    suite (L1 + perceptual + physics), 300 epochs, Optuna 50 trials
  - **Task 41**: `export_onnx.py` — Restormer ONNX export will be added
- ESRGAN polish (tasks 37', 38') will continue in parallel.
- The gate decision is **easily reversible**: if Restormer fails, ESRGAN
  is already production-ready and only the Restormer package needs to be
  disabled.
