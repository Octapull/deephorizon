# infra/docker

Multi-stage Dockerfiles for each service.

| File | Service |
|:---|:---|
| `ml.Dockerfile` | Training image (full `requirements/ml.txt`, CUDA base) |
| `inference.Dockerfile` | Inference server (slim runtime, ONNX model baked in) |
| `training-patch.Dockerfile` | Training image patch (incremental source updates) |
| `api.Dockerfile` | Go API gateway (static binary) |
| `frontend.Dockerfile` | React SPA (build stage + nginx serve) |

Each Dockerfile should use a multi-stage layout: build deps in the first stage, slim runtime in the final stage.

## inference.Dockerfile

Multi-stage build that produces a slim CUDA runtime image with the ONNX
model baked in. Used by the K8s inference deployment.

**Build:**
```bash
docker build -f infra/docker/inference.Dockerfile \
  --build-arg ONNX_MODEL=pix2pix_generator.onnx \
  --build-arg CHECKPOINT_PATH=checkpoints/best_model.pt \
  --build-arg MODEL_TYPE=pix2pix_generator \
  --build-arg INPUT_SHAPE="1,1,256,256" \
  -t deephorizon-inference:v1 .
```

**Run:**
```bash
docker run --gpus all -p 50051:50051 -p 8000:8000 deephorizon-inference:v1
```

**Ports:**
- `50051`: gRPC inference server
- `8000`: Prometheus metrics endpoint (`/metrics`)

**Build args:**
- `ONNX_MODEL`: Output ONNX filename (default: `pix2pix_generator.onnx`)
- `CHECKPOINT_PATH`: Source `.pt` checkpoint (default: `checkpoints/best_model.pt`)
- `MODEL_TYPE`: `unet` or `pix2pix_generator` (default: `pix2pix_generator`)
- `INPUT_SHAPE`: Comma-separated dummy input shape (default: `1,1,256,256`)
