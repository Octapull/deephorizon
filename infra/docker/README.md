# infra/docker

Multi-stage Dockerfiles for each service.

| File | Service |
|:---|:---|
| `ml.Dockerfile` | Inference server (slim runtime, `requirements/serving.txt`) |
| `training.Dockerfile` | Training image (`requirements/ml.txt` on the stock PyTorch CUDA base) — run as a GPU `Job`, see [`docs/runbooks/ML-TRAINING.md`](../../docs/runbooks/ML-TRAINING.md) |
| `api.Dockerfile` | Go API gateway (static binary) |
| `frontend.Dockerfile` | Next.js frontend (standalone Node.js runtime) |

Each Dockerfile should use a multi-stage layout: build deps in the first stage, slim runtime in the final stage.
