# infra

All deployment artifacts: Kubernetes manifests, Dockerfiles, and a local dev compose stack.

**Co-owners:** Stajyer 1, 5, 6 (squad leads). Each squad delivers infra for its own services.

## Layout

| Path | Owner | Contents |
|:---|:---|:---|
| `k8s/data/` | Data squad | Airflow, MinIO manifests (kustomize) |
| `k8s/ml/` | ML squad | MLflow, training Job, inference Deployment |
| `k8s/app/` | Platform squad | Go API, React frontend, Ingress |
| `k8s/monitor/` | Platform squad | Prometheus, Grafana, Argo CD |
| `k8s/secrets/` | Squad leads | SealedSecret manifests (safe to commit) |
| `k8s/app-of-apps.yaml` | Platform | Root Argo CD Application that tracks the four above |
| `docker/` | Per-service | Multi-stage Dockerfiles |
| `docker-compose.dev.yaml` | All | Local stack: MinIO, MLflow, Postgres |

## Apply order (one-time bootstrap)

The team brings these up themselves — see the [References](../README.md#-references) section of the root README for official docs.

1. Install MicroK8s + GPU addon
2. Install Sealed Secrets controller (Helm)
3. Install Argo CD (MicroK8s addon)
4. Apply `k8s/app-of-apps.yaml` once
5. Push manifests to `main` → Argo CD syncs

After bootstrap, deployments happen via `git push` only.
