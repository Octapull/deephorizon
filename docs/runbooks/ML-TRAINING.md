# DeepHorizon remote training runbook

All builds, checks, and training commands run on `10.10.1.132`. Do not copy the
dataset or run Python/Docker training on a workstation.

## Access preflight

From Windows PowerShell:

```powershell
ssh -i "$env:USERPROFILE\.ssh\deephorizon_betul" betul@10.10.1.132
```

Then, on the server:

```bash
kubectl config current-context
kubectl config view --minify -o jsonpath='{..namespace}'; echo
kubectl auth can-i create jobs.batch -n deephorizon-ml
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu
kubectl get pods -n deephorizon-ml
kubectl get secret minio-ml -n deephorizon-ml -o name
```

Never print Secret YAML or decode it. The training Secret must contain the
`ml-team` read-only MinIO account under keys `access_key` and `secret_key`.

## Build on the GPU server

Use a unique image tag; never overwrite a tag already used by a Job.

```bash
docker build -f infra/docker/ml.Dockerfile \
  -t localhost:32000/deephorizon-training:20260810-unet100-v1 .
docker push localhost:32000/deephorizon-training:20260810-unet100-v1
```

## Smoke test

```bash
kubectl apply -f infra/k8s/ml/pvcs.yaml
kubectl apply -f infra/k8s/ml/smoke-training-job.yaml
kubectl get pods -n deephorizon-ml -w
kubectl logs -f -n deephorizon-ml job/unet-smoke-20260810 --timestamps
```

The smoke run must confirm CUDA, 64 MinIO samples, epoch metrics, checkpoint,
and MLflow model/artifact upload before the full run is submitted.

## Full 100-epoch run

```bash
kubectl apply -f infra/k8s/ml/baseline-training-job.yaml
kubectl logs -f -n deephorizon-ml job/unet-baseline-100ep-20260810 --timestamps
```

Each epoch prints its duration, average duration, and maximum remaining ETA.
The Job stops after 15 non-improving validation epochs, but never before epoch
40. Results are visible at `http://10.10.1.132:30500` while connected to VPN.

## Status and failure inspection

```bash
kubectl get job,pod -n deephorizon-ml
kubectl describe job unet-baseline-100ep-20260810 -n deephorizon-ml
kubectl describe pod -n deephorizon-ml <pod-name>
kubectl logs -n deephorizon-ml job/unet-baseline-100ep-20260810 --timestamps
```

To cancel, delete only the named Job. Checkpoints remain on
`training-outputs-pvc`:

```bash
kubectl delete job unet-baseline-100ep-20260810 -n deephorizon-ml
```
