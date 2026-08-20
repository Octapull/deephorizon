# Airflow — Veri Pipeline Orkestrasyonu

DAG'lar veri indirir/üretir, doğrular ve MinIO'ya yükler. Airflow bunları
zamanlar, çalıştırır, izler. Kurulum GitOps ile (`infra/k8s/airflow/`,
`deephorizon-data` namespace).

## Erişim

| | |
|:---|:---|
| **UI (LAN)** | `http://10.10.1.132:30800` |
| **Kullanıcı** | `admin` — parola DevOps'ta (parola kasası) |
| **Tracking (cluster içi)** | `http://airflow-api-server.deephorizon-data.svc:8080` |

UFW 30800'ü yalnızca yerel subnet'e açar. Dışarıdaysan SSH tüneli:
`ssh -L 8080:10.10.1.132:30800 <kullanici>@10.10.1.132` → `http://localhost:8080`

## DAG teslimi — `git push` = deploy

DAG'lar `pipelines/dags/*.py` altında. git-sync repoyu sürekli çeker; `main`'e
push edilen DAG ~30 sn içinde UI'de görünür. Ayrı bir yükleme adımı yok.

## Çalışma ortamı (KONTRAT)

DAG task'ları scheduler pod'unda koşar (LocalExecutor). Ortam şu şekilde hazır:

### Kurulu araçlar (özel imaj)
`infra/docker/airflow.Dockerfile` → stok Airflow + **ehtim, astropy, mc
(MinIO client), dvc[s3], great-expectations**. Task'lar bunları çağırabilir.

### Verilen ortam değişkenleri
Tüm Airflow bileşenlerine (`values.yaml` `env`) veriliyor:

| Değişken | Değer | Anlamı |
|:---|:---|:---|
| `DEEPHORIZON_ROOT` | `/opt/airflow/workspace` | **Yazılabilir** repo kökü (aşağıya bkz.) |
| `MINIO_ENDPOINT` | `http://minio.deephorizon-data.svc:9000` | Cluster içi MinIO |
| `MINIO_USER` | `data-pipeline` | Yükleme kullanıcısı (read-write) |
| `MINIO_ALIAS` | `dh` | `mc` alias adı |

> DAG'lar bu değişkenleri **modül seviyesinde** okuyor (`os.environ[...]`).
> Eksik olsaydı DAG import edilemez, UI'de "Import Error" çıkardı.

### Yazılabilir workspace
git-sync'in getirdiği klon (`/opt/airflow/dags/repo`) **salt-okunur** — script'ler
oraya 20 GB veri yazamaz. Bu yüzden ayrı bir yazılabilir PVC var:

```
/opt/airflow/workspace   (PVC, 50Gi)
   └── init container her pod başlangıcında repoyu buraya klonlar/günceller
        ├── scripts/   ← script'ler buradan koşar
        ├── data/      ← üretilen veri buraya yazılır
        └── .dvc/      ← dvc burada çalışır
```

`DEEPHORIZON_ROOT` buraya işaret eder → script'ler yazılabilir kökten çalışır,
`data/`'ya yazar, `dvc add` çalışır. **Data squad'ın kod değişikliği gerekmez.**

### Parolalar — Airflow Variable (env'de YOK, güvenlik)
DAG'lar iki parolayı Airflow Variable'dan okuyor. İkisi de MinIO kullanıcı parolası,
**DevOps'ta** (kasada) — o yüzden Variable'ları **DevOps tanımlar** (bir kez):

| Variable | Hangi parola | Kullanan |
|:---|:---|:---|
| `minio_data_pipeline_secret` | `data-pipeline` kullanıcısı | upload task'ları (MinIO'ya yazma) |
| `dvc_secret` | `dvc` kullanıcısı | `dvc_track` → `dvc push` (dvc-cache'e) |

```bash
kubectl exec -n deephorizon-data airflow-scheduler-0 -c scheduler -- \
  airflow variables set minio_data_pipeline_secret '<data-pipeline-parolasi>'
kubectl exec -n deephorizon-data airflow-scheduler-0 -c scheduler -- \
  airflow variables set dvc_secret '<dvc-parolasi>'
```

> Eksikse ilgili task düşer: `minio_...` yoksa upload, `dvc_secret` yoksa `dvc push`.
> Airflow yine ayağa kalkar ve DAG'lar görünür.

## DAG'lar

| DAG | Ne yapar | Çıktı |
|:---|:---|:---|
| `eht_ingest` | EHT UVFITS indir → doğrula → yükle | `raw/eht/` |
| `synthetic_generation` | 128×128 sentetik çift üret → doğrula → yükle | `raw/simulated-128/` |
| `training_data_build` | 10K 512×512 çift üret → doğrula → DVC → yükle | `datasets/training-512/` |

Hepsi manuel tetiklenir (`schedule=None`), idempotent. Ayrıntı:
`pipelines/README.md`.

## Durum — uçtan uca test edildi

`synthetic_generation` DAG'ı gerçek veriyle baştan sona koşuldu (`state=success`):
üret → doğrula (GE) → MinIO'ya yükle → `dvc add` → `dvc push` (2021 dosya dvc-cache'e).
Yani özel imaj, workspace, iki Variable ve DVC zinciri çalışır durumda.

**Yeni Airflow için kurulum tekrarı gerekirse tek yapılacak:** iki Variable'ı set etmek
(yukarıdaki komutlar). Gerisi GitOps'tan gelir.

## Sınırlar / notlar

- **LocalExecutor** — task'lar scheduler pod'unda koşar. Tek node için uygun;
  ağır iş yükünde ayrı worker (Celery/KubernetesExecutor) gerekebilir.
- **20 GB üretim** scheduler pod'unu yorar. `training_data_build` uzun sürer;
  workspace PVC 50Gi (veri MinIO'ya yüklenince temizlenebilir, kalıcı depo MinIO).
- **git-sync klonu vs workspace** — DAG dosyaları git-sync'ten (canlı güncelleme),
  script'ler workspace klonundan (pod restart'ta güncellenir) gelir. Script
  değişikliği için scheduler restart gerekir.

## DevOps notları

| | |
|:---|:---|
| Manifest kaynağı | `infra/k8s/airflow/values.yaml` (render kaynağı) |
| Uygulanan | `infra/k8s/airflow/manifests/airflow.yaml` (rendered) |
| Argo CD app | `infra/k8s/apps/airflow.yaml` |
| Postgres | `infra/k8s/postgresql/airflow/postgres.yaml` |
| Workspace PVC | `infra/k8s/airflow/workspace-pvc.yaml` |
| Özel imaj | `infra/docker/airflow.Dockerfile` → `localhost:32000/deephorizon-airflow` |
| Secret'lar | `airflow-db-credentials`, `airflow-keys` (SealedSecret) |
| Log | `kubectl logs -n deephorizon-data airflow-scheduler-0 -c scheduler` |

**Rendered manifests:** Argo CD'nin Helm'i (v3.11) chart'ı render edemediği için
Helm'i biz çalıştırıp çıktıyı repoya koyuyoruz. `values.yaml`'ı değiştirdiysen
yeniden render alıp `manifests/airflow.yaml`'ı da commit et — komut
`values.yaml` başında.

**İmaj güncelleme:** Yeni imaj = yeni tarih etiketi (`latest` kullanma — Kubernetes
aynı etiketi yeniden çekmez). Build + push sonrası `values.yaml`'da
`images.airflow.tag` güncelle → yeniden render.
