# infra/docker

Multi-stage Dockerfiles for each service.

| File | Service |
|:---|:---|
| `ml.Dockerfile` | Inference server (slim runtime, `requirements/serving.txt`) |
| `training.Dockerfile` | Training image (full `requirements/ml.txt`, CUDA base) |
| `api.Dockerfile` | Go API gateway (static binary) |
| `frontend.Dockerfile` | React SPA (build stage + nginx serve) |

Each Dockerfile should use a multi-stage layout: build deps in the first stage, slim runtime in the final stage.
