# Runbook — GPU Sunucuda Model Eğitimi

ML ekibinin `deephorizon-ml` namespace'inde Kubernetes Job açarak L40S üzerinde
eğitim koşturmasının tam akışı: kimin neyi bir kez yaptığı, ML tarafının günlük
döngüsü, ve tıkandığı yerde nereye bakacağı.

**Temel kural:** eğitim `python train.py` ile SSH oturumunda koşmaz. GPU'yu
Kubernetes dağıtır; işi isteyen bir **Job** açar, pod GPU'lu node'a yerleşir,
iş bitince GPU serbest kalır. Kazancı: SSH kopsa da eğitim devam eder, çöken
koşu yeniden denenir, GPU'yu iki kişi aynı anda kapamaz.

## Sorumluluk sınırı

| İş | Sahibi | Sıklık |
|:---|:---|:---|
| Eğitim imajını build + push | DevOps | Kod veya bağımlılık değişince |
| MinIO credential Secret'ı | DevOps | Bir kez (+ rotasyonda) |
| RBAC / kubeconfig | DevOps | Kişi eklendiğinde |
| Job manifest'i yazma, koşuyu başlatma/izleme | ML squad | Her koşuda |

ML tarafının imaj build etmesi **gerekmez ve beklenmez** — Docker grubuna üyelik
host'ta root yetkisine eşdeğerdir, bu yüzden verilmez.

---

## Bölüm A — Bir kerelik kurulum (DevOps)

### A1. Eğitim imajını build et ve push'la

Sunucuda, repo kökünden:

```bash
TAG=$(date +%F)          # ornek: 2026-08-07
docker build -f infra/docker/training.Dockerfile \
  -t localhost:32000/deephorizon-training:$TAG .
docker push localhost:32000/deephorizon-training:$TAG
```

- `localhost:32000` MicroK8s'in yerleşik registry'si (`container-registry`
  namespace'i). Cluster imajı oradan çeker; harici bir registry gerekmez.
- **`latest` etiketi kullanma.** Kubernetes aynı etiketi yeniden çekmez;
  imajı güncellersin, Job eski katmanla koşar ve fark günlerce görülmez.
- Eğitim kodu şu an `ml/feature` dalında, Dockerfile `main`'de. Merge etmeye
  gerek yok — kodu `ml/feature`'dan al, üç dosyayı `main`'den üzerine giydir:
  ```bash
  git fetch origin
  git checkout -B build/training origin/ml/feature
  git checkout origin/main -- \
    infra/docker/training.Dockerfile \
    infra/docker/training.Dockerfile.dockerignore \
    requirements/ml.txt
  ```
  **İkinci `checkout`'ta `--` ve dosya listesini düşürme.** `git checkout
  origin/main` (dosya listesi olmadan) tüm ağacı `main`'e çevirir; build
  sessizce `main`'in `services/ml`'iyle yapılır ve ilk koşu `No module named
  services.ml.training.train` ile düşer. Build'den önce doğrula:
  ```bash
  ls services/ml/training/train.py services/ml/conf/config.yaml
  ```

Doğrulama — imaj cluster'dan çekilebiliyor mu:

```bash
kubectl -n deephorizon-ml run img-test --rm -it --restart=Never \
  --image=localhost:32000/deephorizon-training:$TAG -- python -c "import torch; print(torch.__version__)"
```

### A2. MinIO credential Secret'ı

Eğitim kodu veriyi MinIO'dan `boto3` ile çeker ve üç env bekler:
`MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`.

Secret adı **`minio-ml`**, anahtarlar **`access_key`** / **`secret_key`**
(alt çizgi). Job manifest'leri bu isimlendirmeyi bekliyor.

Kullanıcı: **`ml-trainer`** — `raw` + `datasets` bucket'larında yalnızca okuma
yetkisi olan, yalnızca eğitim için açılmış hesap (`ml-read` policy'si,
bkz. `docs/DATA.md`). Root credential **kullanılmaz**.

> **Neden `ml-team` değil:** `ml-team` insanların da `mc`/`boto3` ile kullandığı
> bir hesap. Parolası döndüğü gün eğitim Job'ları sessizce kırılır ve kimse
> sebebini bu değişiklikle ilişkilendiremez. Servis kimliği insan kimliğinden
> ayrı tutulur; böylece ikisi bağımsız rotasyona girer.

```bash
kubectl create secret generic minio-ml -n deephorizon-ml \
  --from-literal=access_key='ml-trainer' \
  --from-literal=secret_key='<ml-trainer-parolasi>' \
  --dry-run=client -o yaml \
  | kubeseal -n deephorizon-ml -o yaml \
  | kubectl apply -f -
```

SealedSecret YAML'ı **Git'e girmez** (proje kuralı, `infra/k8s/secrets/`
klasörleri boş). Şifreleme namespace adına scope'ludur — `-n deephorizon-ml`
şart, başka namespace'te çözülmez.

Doğrulama — Secret'ın varlığı ve içindeki kullanıcı:
```bash
kubectl -n deephorizon-ml get secret minio-ml
kubectl -n deephorizon-ml get secret minio-ml -o jsonpath='{.data.access_key}' | base64 -d; echo
```

Credential'ın kendisini MinIO'ya karşı doğrulamak için üçlü test
(`docs/DEVOPS.md` Hata #8'in dersi — okuma çalışmalı, yazma ve kapsam dışı
reddedilmeli):
```bash
sudo -u deephorizon mc alias set mltest http://10.10.1.132:30900 ml-trainer '<parola>'
sudo -u deephorizon mc ls   mltest/datasets/training-512/v1/   # calismali
sudo -u deephorizon mc cp   /etc/hostname mltest/datasets/x    # reddedilmeli
sudo -u deephorizon mc ls   mltest/mlflow/                     # reddedilmeli
sudo -u deephorizon mc alias rm mltest
```

### A3. Çıktı diski

Eğitim Job'ları `/app/runs` altına `training-outputs-pvc` bağlar (checkpoint'ler,
Hydra çıktıları, örnek PNG'ler). PVC GitOps'ta:
`infra/k8s/ml-training/pvcs.yaml` → `apps/ml-training.yaml` Application'ı.

Elle bir şey yapmaya gerek yok; Argo CD oluşturur. `Prune=false` anotasyonu
taşır — manifest repodan kalksa bile PVC silinmez, çünkü silinmesi süren bir
koşunun tüm ara durumunu götürür.

> **Egitim Job'ları GitOps'a girmez.** Her koşu ad-hoc'tur, kendi adı ve
> hyperparametreleriyle elle uygulanır. Job'ları Argo CD'ye verirsen biten
> koşuları yeniden oluşturur ya da süren bir koşuyu prune eder.

```bash
kubectl -n deephorizon-ml get pvc training-outputs-pvc
```

### A4. Erişim (kişi eklendiğinde)

Kişiye sunucuda yetkisiz bir Unix kullanıcısı + SSH key açılır ve
`~/.kube/config` olarak `ml-trainer` ServiceAccount token'ı kurulur. Bu SA
`deephorizon-ml` namespace'ine scoped: Job açabilir/silebilir, pod ve log
okuyabilir, PVC/ConfigMap/Secret yönetebilir; namespace dışına çıkamaz.

> RBAC 2026-08-07'de etkinleştirildi. Öncesinde cluster `AlwaysAllow` ile
> çalışıyordu ve yazılı RBAC uygulanmıyordu — bkz. `docs/DEVOPS.md`.

---

## Bölüm B — Eğitim koşusu (ML squad)

### B1. Job manifest'ini hazırla

Şablon: [`training-job.example.yaml`](training-job.example.yaml)

```bash
cp docs/runbooks/training-job.example.yaml ~/unet-01.yaml
nano ~/unet-01.yaml
```

Her koşuda değiştirilecek iki yer:

| Alan | Ne yazılır |
|:---|:---|
| `metadata.name` | Benzersiz ad (`unet-baseline-01`, `unet-lr1e4-02`…). Aynı adla ikinci Job açılamaz. |
| `args` | Hydra override'ları: `training.epochs=50`, `training.learning_rate=0.0001`, `loss.name=l1` … |

Hyperparametrelerin varsayılanları `services/ml/conf/` altında. `args` boş
bırakılırsa varsayılanlarla koşar.

### B2. Başlat

```bash
kubectl apply -f ~/unet-01.yaml
```

### B3. İzle

```bash
kubectl get jobs
kubectl get pods -w                       # Pending -> ContainerCreating -> Running
kubectl logs -f job/unet-baseline-01      # canli log
```

`kubectl logs -f` kesilirse eğitim etkilenmez, tekrar bağlanılır. SSH oturumu
kapansa da Job cluster'da koşmaya devam eder.

### B4. Sonuçlar

Her koşu MLflow'a loglanır: tüm config parametre olarak, epoch metrikleri,
`best_model.pt` ve örnek PNG'ler artifact olarak.

- MLflow UI (LAN): `http://10.10.1.132:30500`
- Cluster içi: `http://mlflow.deephorizon-ml.svc:5000`

### B5. Bitir / iptal et

```bash
kubectl delete job unet-baseline-01
```

Job `ttlSecondsAfterFinished: 86400` ile 24 saat sonra kendini siler. Metrikler
ve model MLflow'da kalır — pod'un silinmesi sonuçları kaybettirmez.

---

## Sorun giderme

| Belirti | Sebep | Ne yapmalı |
|:---|:---|:---|
| Pod `Pending`, uzun süre başlamıyor | GPU meşgul — sunucuda **tek L40S** var, başka bir Job onu tutuyor | `kubectl describe pod <ad>` → `Insufficient nvidia.com/gpu`. Diğer koşunun bitmesini bekle ya da sahibiyle konuş |
| `ImagePullBackOff` | Etiket yanlış ya da imaj push'lanmamış | Manifest'teki etiketi DevOps'un push'ladığıyla karşılaştır |
| `CreateContainerConfigError` | `minio-ml` Secret'ı yok ya da anahtar adları tutmuyor (`access_key`/`secret_key`) | DevOps → A2 |
| `ModuleNotFoundError` | İmaj eski, yeni bağımlılık eklenmiş | DevOps yeni etiketle build+push eder |
| `StartError` + `exec: "training.epochs=1": executable file not found` | Manifest'te `command` yazılmış, ENTRYPOINT ezilmiş | Manifest'ten `command` satırını sil; `args` tek başına yeterli |
| `No module named services.ml.training.train` | İmaj yanlış daldan build edilmiş (`main`'in `services/ml`'i girmiş) | DevOps → A1'deki checkout doğrulaması |
| `Bus error` / DataLoader worker çöküyor | `/dev/shm` varsayılanı 64 MB | Şablondaki `dshm` volume'u manifest'te duruyor mu kontrol et; yoksa `data.num_workers=0` ile geç |
| `CUDA out of memory` | Batch büyük | `training.batch_size` düşür, `data.crop_size` küçült ya da `training.amp=true` (L40S bf16 destekler) |
| MinIO'dan dosya listelenmiyor, 0 örnek | Prefix yanlış (aşağıdaki nota bak) | `mc ls` ile gerçek yolu doğrula, `data.minio_prefix` override'la |
| `Forbidden` | Namespace dışına çıkılmaya çalışıldı | Yetki `deephorizon-ml` ile sınırlı; ihtiyaç varsa DevOps'a yaz |
| Eğitim bitti ama artifact yüklenmiyor: `RemoteDisconnected` → `Connection refused` | MLflow artifact yüklerken OOMKilled oldu ve CrashLoopBackOff'a girdi | **Önce Job'ı sil** — istemci yeniden denedikçe MLflow'u tekrar öldürür ve döngü kapanmaz. Sonra `kubectl -n deephorizon-ml describe pod -l app=mlflow \| grep -A5 "Last State"` ile teyit et, DevOps limiti yükseltir |

Pod'un neden başlamadığını anlamanın tek adresi:
```bash
kubectl describe pod <pod-adi>       # en alttaki Events bolumu
```

---

## Bilinen açıklar / dikkat

- **MinIO prefix'i yanlış (doğrulandı, 2026-08-07).**
  `services/ml/conf/data/default.yaml` `bucket_name: datasets` **ve**
  `minio_prefix: datasets/training-512/v1` diyor. Sunucudaki gerçek düzen:
  ```
  $ mc ls dh/datasets/training-512/v1/
  clean/   degraded/
  ```
  Yani prefix `training-512/v1` olmalı; mevcut haliyle boto3
  `datasets/datasets/training-512/v1` arar ve sıfır dosya bulur. Config
  düzeltilene kadar her koşuya override eklenmeli:
  ```
  args: ["data.minio_prefix=training-512/v1"]
  ```
  Kalıcı düzeltme ML squad'da.
- **Tek GPU.** İkinci Job `Pending` bekler; bu doğru davranış. Inference
  deploy edildiğinde kök README'deki "training öncesi inference `replicas: 0`"
  politikası devreye girecek.
- **`lpips` ilk kullanımda ağdan ağırlık indirir.** Perceptual loss'a geçen
  koşularda pod'un dışarı erişimi olmalı; kapalı ortamda ağırlıklar imaja
  gömülmeli.
- **Python sürümü.** İmaj Python 3.11 (stok PyTorch CUDA imajı); repo kökü
  3.13 istiyor. Fark bilinçli — 3.13 alt sınırı `ehtim` için, eğitim kodu onu
  kullanmıyor. Gerekçe `infra/docker/training.Dockerfile` başlığında.
- **numpy sürümü.** İmaj `requirements/base.txt` uyarınca numpy 2.x kurar;
  `ml/feature` dalının `pyproject.toml`'u Intel Mac uyumluluğu için
  `numpy<2.0` pinliyor. İkisi farklı ortamlar (konteyner vs. lokal venv) ama
  ilk koşuda numpy kaynaklı bir hata çıkarsa ilk bakılacak yer burası.
