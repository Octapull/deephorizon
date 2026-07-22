# infra

All deployment artifacts: Kubernetes manifests, Dockerfiles, and a local dev compose stack.

**Co-owners:** Stajyer 1, 5, 6 (squad leads). Each squad delivers infra for its own services.

## Layout

| Path | Owner | Contents |
|:---|:---|:---|
| `k8s/airflow/` | Data squad | Airflow manifests, render values, workspace PVC |
| `k8s/minio/` | Data squad | MinIO StatefulSet and Services |
| `k8s/mlflow/` | ML squad | MLflow Deployment and Service |
| `k8s/postgresql/` | Data / ML squads | Airflow and MLflow PostgreSQL instances |
| `k8s/redis/` | Platform squad | Redis Deployment and Service |
| `k8s/monitor/` | Platform squad | Prometheus, Grafana, Argo CD |
| `k8s/secrets/` | Squad leads | Empty service placeholders; Secret YAML files stay outside Git |
| `k8s/apps/` | Platform | Technology-level Argo CD `Application` manifests tracked by the root app |
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
`k8s/<technology>/`, add an Application in `k8s/apps/` if the technology is new,
and open a PR. Secret YAML files are managed outside Git and must not be committed.
