# infra

All deployment artifacts: Kubernetes manifests, Dockerfiles, and a local dev compose stack.

**Co-owners:** Stajyer 1, 5, 6 (squad leads). Each squad delivers infra for its own services.

## Layout

| Path | Owner | Contents |
|:---|:---|:---|
| `k8s/data/` | Data squad | Airflow, MinIO manifests (kustomize) |
| `k8s/ml/` | ML squad | MLflow, training Job, inference Deployment |
| `k8s/app/` | Platform squad | Go API + Next.js frontend (NodePort Service only — no cluster Ingress; NGINX Proxy Manager handles TLS on the host) |
| `k8s/monitor/` | Platform squad | Prometheus, Grafana, Argo CD |
| `k8s/secrets/` | Squad leads | SealedSecret manifests (safe to commit) |
| `k8s/apps/` | Platform | Squad-level Argo CD `Application` manifests (one per area) tracked by the root app |
| `k8s/app-of-apps.yaml` | Platform | Root Argo CD Application that tracks `k8s/apps/` |
| `docker/` | Per-service | Multi-stage Dockerfiles |
| `docker-compose.dev.yaml` | All | Local stack: MinIO, MLflow, Postgres |

## Apply order (one-time bootstrap)

> **Status: DONE (2026-07-14).** All five steps completed and field-verified; the full
> log — including every error hit, its cause, and the fix — is in
> [`docs/DEVOPS.md`](../docs/DEVOPS.md). First workload (MinIO) is live under
> `deephorizon-data`; data layout and access guide in [`docs/DATA.md`](../docs/DATA.md).

1. Install MicroK8s + GPU addon
2. Install Sealed Secrets controller (Helm)
3. Install Argo CD (MicroK8s addon)
4. Apply `k8s/app-of-apps.yaml` once
5. Push manifests to `main` → Argo CD syncs

After bootstrap, deployments happen via `git push` only: add manifests under
`k8s/<area>/`, add an Application in `k8s/apps/` if the area is new, seal secrets
into `k8s/secrets/`, open a PR.
