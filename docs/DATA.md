# Veri Deposu (MinIO) — Düzen ve Erişim

Tüm üretilen/indirilen veri, sunucudaki MinIO'da yaşar. Lokal `data/` klasörü geçici
çalışma alanıdır (`.gitignore`'da); **kalıcı kaynak MinIO'dur**.

## Bucket düzeni

```
raw/                          # DEĞİŞMEZ kaynak veri — bir kez yüklenir, üzerine yazılmaz
├── eht/<dataset>/*.uvfits    #   gerçek EHT gözlemleri (scripts/download_eht_data.py)
└── simulated-128/v1/         #   ehtim tabanlı 128x128 set (scripts/generate_synthetic_data.py)
    ├── clean/     *.npy
    ├── degraded/  *.npy
    └── pairs/     *.png      #   göz kontrolü için önizlemeler

datasets/                     # eğitim setleri — ML ekibinin ana adresi
└── training-512/v1/          #   10k çift 512x512, ~20 GiB (scripts/generate_training_data.py)
    ├── clean/     00000.npy …
    └── degraded/  00000.npy …

mlflow/                       # rezerve — MLflow artifact store (kurulunca)
```

**Versiyonlama kuralı:** üretim parametreleri (PSF, gürültü, model dağılımı…) değişirse
aynı prefix'e yazılmaz — `v2/` açılır. Böylece eski deneyler hangi veriyle koştuğunu
kaybetmez. (DVC entegrasyonu bu düzenin üstüne gelecek.)

## Erişim bilgileri

| | |
|:---|:---|
| **S3 endpoint (LAN)** | `http://10.10.1.132:30900` |
| **S3 endpoint (cluster içi)** | `http://minio.deephorizon-data.svc:9000` |
| **Console (tarayıcı)** | `http://10.10.1.132:30901` (LAN) — bucket gezinme |
| **Kullanıcı: `ml-team`** | özel `ml-read` policy: listeleme+okuma, yalnızca `raw`+`datasets` — model eğitimi/inceleme için bunu kullan |
| **Kullanıcı: `data-pipeline`** | read-write — yalnızca veri üreten kişi/pipeline kullanır |

Parolalar DevOps'tan (takım parola kasası). **Root credential kimseyle paylaşılmaz.**

## ML tarafı: veri nasıl çekilir

**mc ile** (hızlı, tavsiye edilen):

```bash
mc alias set dh http://10.10.1.132:30900 ml-team <parola>
mc ls dh/datasets/training-512/v1/
mc mirror dh/datasets/training-512/v1/ ./data/training/   # lokale indir
```

**Python (PyTorch Dataset içinden, s3fs ile):**

```python
import s3fs, numpy as np

fs = s3fs.S3FileSystem(
    key="ml-team", secret="<parola>",
    endpoint_url="http://10.10.1.132:30900",
)
with fs.open("datasets/training-512/v1/clean/00000.npy") as f:
    img = np.load(f)
```

## Veri üretimi: yükleme akışı (data-pipeline rolü)

Scriptler sunucuda çalıştırılır, çıktı `mc mirror` ile yüklenir:

```bash
python scripts/download_eht_data.py
mc mirror data/raw/eht/ dh/raw/eht/

python scripts/generate_synthetic_data.py
mc mirror data/raw/simulated/ dh/raw/simulated-128/v1/

python scripts/generate_training_data.py
mc mirror data/training/ dh/datasets/training-512/v1/
```

> Not: `training-512` dosya adlarında degradasyon seviyesi yok (`00000.npy`). Seviye
> bazlı filtreleme gerekirse üretim scriptine bir `manifest.csv` eklenmesi ML+Data
> squad'ın konusu.
