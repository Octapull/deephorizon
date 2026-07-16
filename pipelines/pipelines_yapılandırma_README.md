# Pipelines
16.07.2026

Apache Airflow DAG'ları ile DeepHorizon veri pipeline'ı.

## Genel Bakış

Bu klasör üç Airflow DAG'ı içerir. Her DAG manuel tetiklenir (`schedule=None`).
Tüm DAG'lar idempotent'tir — aynı DAG birden fazla çalıştırılabilir.

---

## DAG'lar

### 1. `eht_ingest`

**Ne yapar:** Kamuya açık EHT (Event Horizon Telescope) UVFITS gözlem dosyalarını GitHub'dan indirir, doğrular ve MinIO'ya yükler.

**Task zinciri:**
```
download_eht → validate_eht → upload_to_minio → dvc_track
```

| Task | Açıklama |
|:---|:---|
| `download_eht` | 7 EHT dataset'ini GitHub'dan indirir (`scripts/download_eht_data.py`) |
| `validate_eht` | İndirilen dosyaların boyutunu kontrol eder; bozuk dosya varsa hata verir |
| `upload_to_minio` | `raw/eht/<dataset>/` path'ine yükler |
| `dvc_track` | `data/raw/eht` dizinini DVC ile izlemeye alır |

**Çıktı:** `MinIO → raw/eht/`

---

### 2. `synthetic_generation`

**Ne yapar:** `eht-imaging` kütüphanesi ile fiziğe dayalı 128×128 clean/degraded görüntü çiftleri üretir. Hızlı model mimarisi deneyleri için kullanılır (1K çift, ~100 MB).

**Task zinciri:**
```
check_ehtim → generate_synthetic → validate_synthetic → upload_to_minio → dvc_track
```

| Task | Açıklama |
|:---|:---|
| `check_ehtim` | `ehtim` kütüphanesinin kurulu olduğunu doğrular |
| `generate_synthetic` | 1K adet 128×128 çift üretir (`scripts/generate_synthetic_data.py`) |
| `validate_synthetic` | GE ile shape, dtype, NaN, negatif değer kontrolü yapar |
| `upload_to_minio` | `raw/simulated-128/<timestamp>/` path'ine yükler |
| `dvc_track` | `data/raw/simulated` dizinini DVC ile izlemeye alır |

**Çıktı:** `MinIO → raw/simulated-128/<timestamp>/`

**Model dağılımı:** Crescent %70, Ring %30

**Bozulma seviyeleri:** light, medium, heavy, extreme

---

### 3. `training_data_build`

**Ne yapar:** 10K adet 512×512 clean/degraded görüntü çifti üretir, Great Expectations ile doğrular, DVC ile versiyonlar ve MinIO'ya yükler. ML ekibinin model eğitiminde kullandığı ana dataset budur.

**Task zinciri:**
```
generate_training → validate_training → dvc_track → upload_to_minio
```

| Task | Açıklama |
|:---|:---|
| `generate_training` | 10K adet 512×512 çift üretir (`scripts/generate_training_data.py`) |
| `validate_training` | GE ile shape, dtype, NaN, negatif değer kontrolü yapar (500 örnekle) |
| `dvc_track` | `data/training` dizinini DVC ile izlemeye alır |
| `upload_to_minio` | `datasets/training-512/<timestamp>/` path'ine yükler |

**Çıktı:** `MinIO → datasets/training-512/<timestamp>/`

**Model dağılımı:** Crescent %60, Ring %25, Double Ring %15

**Bozulma seviyeleri:** light, medium, heavy, extreme

**Veri boyutu:** ~20 GB

---

## Ortak Özellikler

**Güvenlik:** MinIO credentials hiçbir zaman subprocess argümanında görünmez.
`MC_HOST_<alias>` environment variable pattern'i kullanılır.

**DVC:** Her DAG `dvc add` yapar ancak `dvc push` yapmaz.
DVC remote yapılandırıldıktan sonra `dvc_track` task'larına `dvc push` eklenebilir.
`git commit` hiçbir DAG'ın sorumluluğu değildir.

**Validation:** `scripts/ge_suite.py` mevcutsa Great Expectations kullanılır.
Yoksa lightweight numpy fallback devreye girer.

---

## Zorunlu Environment Variable'lar

DAG'larda hiçbir değer hardcoded değildir. Aşağıdaki 5 variable
tanımlı olmadan DAG'lar çalışmaz — eksikse `KeyError` fırlatır.

| Variable | Açıklama | Örnek |
|:---|:---|:---|
| `DEEPHORIZON_ROOT` | Repo'nun sunucudaki tam path'i | `/home/<kullanıcı>/deephorizon` |
| `MINIO_ENDPOINT` | MinIO S3 endpoint URL'i | `http://10.10.1.132:30900` |
| `MINIO_USER` | MinIO kullanıcı adı | `data-pipeline` |
| `MINIO_ALIAS` | mc komutunda kullanılan alias | `dh` |
| `MINIO_SECRET` | MinIO kullanıcı parolası | DevOps'tan alınır |

Airflow worker'ında tanımlanması:

```bash
export DEEPHORIZON_ROOT=/home/<kullanıcı>/deephorizon
export MINIO_ENDPOINT=http://10.10.1.132:30900
export MINIO_USER=data-pipeline
export MINIO_ALIAS=dh
export MINIO_SECRET=<parola>
```

Parolayı Airflow Variable olarak da tanımlayabilirsin:

```bash
airflow variables set minio_data_pipeline_secret <parola>
```

---

## Kurulum

```bash


# 1. Environment variable'ları tanımla (yukarıya bak)

# 2. Airflow DAG klasörünü ayarla
export AIRFLOW__CORE__DAGS_FOLDER=/path/to/deephorizon/pipelines/dags


```

## DAG Çalıştırma

CLI üzerinden:
```bash
airflow dags test eht_ingest 2026-07-16
airflow dags test synthetic_generation 2026-07-16
airflow dags test training_data_build 2026-07-16
```

## Çalışma Sırası

DAG'lar birbirinden bağımsızdır, herhangi bir sırada çalışabilir.
Önerilen sıra:

```
1. eht_ingest          → gerçek EHT verisi (küçük, hızlı)
2. synthetic_generation → hızlı prototipleme seti (~100 MB)
3. training_data_build  → ana eğitim seti (~20 GB, uzun sürer)
```

## .gitignore Notu

DVC `.dvc` dosyalarını `data/` altına yazar. Bu dosyaların git'e girmesi
için `.gitignore` şu şekilde yapılandırılmalıdır:

```gitignore
# Büyük veri klasörleri (DVC yönetir)
data/training/
data/raw/eht/
data/raw/simulated/

# .dvc dosyaları git'e girer — buraya ekleme yapma
```

`data/` klasörünün tamamını ignore etme — DVC `.dvc` dosyalarını göremez.

## Kişisel not

DVC ve Airflow sunucuya kurulduğunda kodlar tam anlamıyla çalışacaktır şu an localde test edebilirsiniz.
