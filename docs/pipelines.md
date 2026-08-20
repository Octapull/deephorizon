# Pipelines

Apache Airflow DAG'ları ile DeepHorizon veri pipeline'ı.

## Genel Bakış

Bu klasör üç Airflow DAG'ı içerir. Her DAG manuel tetiklenir (`schedule=None`). Tüm DAG'lar idempotent'tir — aynı DAG birden fazla çalıştırılabilir.

---

## DAG'lar

### 1. `eht_ingest`

Kamuya açık EHT (Event Horizon Telescope) UVFITS gözlem dosyalarını GitHub'dan indirir, doğrular, MinIO'ya yükler ve DVC ile izlemeye alır.

**Task zinciri:** `download_eht → validate_eht → upload_to_minio → dvc_track`

| Task | Açıklama |
|:---|:---|
| `download_eht` | 7 EHT dataset'ini GitHub'dan indirir (`scripts/download_eht_data.py`) |
| `validate_eht` | Dosya sayısı ve boyut kontrolü; GE mevcutsa ek doğrulama |
| `upload_to_minio` | `raw/eht/<dataset>/` path'ine yükler |
| `dvc_track` | `data/raw/eht` dizinini DVC'ye ekler ve `dvc-cache` bucket'ına push'lar |

**Çıktı:** MinIO → `raw/eht/`

---

### 2. `synthetic_generation`

eht-imaging kütüphanesi ile fiziğe dayalı 128×128 clean/degraded görüntü çiftleri üretir. Hızlı model mimarisi deneyleri için kullanılır (1K çift, ~100 MB).

**Task zinciri:** `check_ehtim → generate_synthetic → validate_synthetic → upload_to_minio → dvc_track`

| Task | Açıklama |
|:---|:---|
| `check_ehtim` | ehtim kütüphanesinin kurulu olduğunu doğrular |
| `generate_synthetic` | 1K adet 128×128 çift üretir (`scripts/generate_synthetic_data.py`) |
| `validate_synthetic` | GE ile shape, dtype, NaN, negatif değer kontrolü yapar |
| `upload_to_minio` | `raw/simulated-128/<timestamp>/` path'ine yükler |
| `dvc_track` | `data/raw/simulated` dizinini DVC'ye ekler ve push'lar |

**Çıktı:** MinIO → `raw/simulated-128/<timestamp>/`

Model dağılımı: Crescent %70, Ring %30 · Bozulma seviyeleri: light, medium, heavy, extreme

---

### 3. `training_data_build`

10K adet 512×512 clean/degraded görüntü çifti üretir, Great Expectations ile doğrular, DVC ile versiyonlar ve MinIO'ya yükler. ML ekibinin model eğitiminde kullandığı ana dataset budur.

**Task zinciri:** `generate_training → validate_training → dvc_track → upload_to_minio`

| Task | Açıklama |
|:---|:---|
| `generate_training` | 10K adet 512×512 çift üretir (`scripts/generate_training_data.py`) |
| `validate_training` | GE ile shape, dtype, NaN, negatif değer kontrolü yapar (500 örnekle) |
| `dvc_track` | `data/training` dizinini DVC'ye ekler ve `dvc-cache` bucket'ına push'lar |
| `upload_to_minio` | `datasets/training-512/<timestamp>/` path'ine yükler |

**Çıktı:** MinIO → `datasets/training-512/<timestamp>/`

Model dağılımı: Crescent %60, Ring %25, Double Ring %15 · Bozulma seviyeleri: light, medium, heavy, extreme · Veri boyutu: ~20 GB

---

## Ortak Özellikler

**Güvenlik:** MinIO ve DVC credential'ları hiçbir zaman subprocess argümanında görünmez.
MinIO için `MC_HOST_<alias>` env pattern'i, DVC için `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` kullanılır.

**Validation:** `scripts/ge_suite.py` mevcutsa Great Expectations kullanılır. Yoksa lightweight numpy fallback devreye girer.

**DVC:** Her DAG `dvc add` + `dvc push` yapar. `git commit` hiçbir DAG'ın sorumluluğu değildir — push sonrası `.dvc` pointer dosyasını elle commit etmen gerekir.

---

## Zorunlu Environment Variable'lar

DAG'larda hiçbir değer hardcoded değildir. Aşağıdaki variable'lar tanımlı olmadan DAG'lar çalışmaz.

| Variable | Açıklama | Örnek |
|:---|:---|:---|
| `DEEPHORIZON_ROOT` | Repo'nun sunucudaki tam path'i | `/home/deephorizon/deephorizon` |
| `MINIO_ENDPOINT` | MinIO S3 endpoint URL'i | `http://10.10.1.132:30900` |
| `MINIO_USER` | MinIO kullanıcı adı | `data-pipeline` |
| `MINIO_ALIAS` | `mc` komutunda kullanılan alias | `dh` |
| `MINIO_SECRET` | MinIO kullanıcı parolası | Airflow Variable veya env |
| `DVC_SECRET` | `dvc` kullanıcısının MinIO parolası | Airflow Variable veya env |

Airflow worker'ında tanımlanması:

```bash
export DEEPHORIZON_ROOT=/home/deephorizon/deephorizon
export MINIO_ENDPOINT=http://10.10.1.132:30900
export MINIO_USER=data-pipeline
export MINIO_ALIAS=dh
```

Parolaları Airflow Variable olarak tanımla (önerilir):

```bash
airflow variables set minio_data_pipeline_secret <parola>
airflow variables set dvc_secret <parola>
```

Ya da env üzerinden:

```bash
export MINIO_SECRET=<parola>
export DVC_SECRET=<parola>
```

---

## Kurulum

```bash
# 1. Bağımlılıkları kur
uv sync --extra data

# 2. Environment variable'ları tanımla (yukarıya bak)

# 3. Airflow DAG ve plugin klasörlerini ayarla
export AIRFLOW__CORE__DAGS_FOLDER=/path/to/deephorizon/pipelines/dags
export AIRFLOW__CORE__PLUGINS_FOLDER=/path/to/deephorizon/pipelines/plugins
```

---

## DVC Kurulum (bir kez, repo başına)

`.dvc/config` dosyası repo'da mevcut — remote URL ve bucket adı zaten tanımlı. Sadece lokal credential'ları ayarla:

```bash
dvc init   # .dvc/ klasörü yoksa

dvc remote modify --local minio access_key_id dvc
dvc remote modify --local minio secret_access_key '<parola>'  # DevOps'tan alınır
```

> 🔴 `--local` zorunlu — onsuz parola `.dvc/config`'e yazılır ve Git'e commit edilir.

Sonra:

```bash
git add .dvc/config .dvc/.gitignore .dvcignore
git commit -m "chore(data): configure DVC remote (MinIO dvc-cache)"
git push
```

Detaylı kurulum: [`docs/DVC.md`](../docs/DVC.md)

---

## DAG Çalıştırma

Tek task test etmek için:

```bash
airflow tasks test eht_ingest validate_eht 2026-01-01
airflow tasks test synthetic_generation validate_synthetic 2026-01-01
airflow tasks test training_data_build validate_training 2026-01-01
```

Tüm DAG'ı çalıştırmak için Airflow UI → DAG → Trigger.

## Çalışma Sırası

DAG'lar birbirinden bağımsızdır. Önerilen sıra:

```
1. eht_ingest           → gerçek EHT verisi (küçük, hızlı)
2. synthetic_generation → hızlı prototipleme seti (~100 MB)
3. training_data_build  → ana eğitim seti (~20 GB, uzun sürer)
```