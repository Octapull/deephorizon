# DeepHorizon

> Deep learning-based super-resolution and denoising pipeline for black hole images captured by radio telescope arrays

---

## İçindekiler

- [Proje Özeti](#proje-özeti)
- [Problem Tanımı](#problem-tanımı)
- [Mimari Genel Bakış](#mimari-genel-bakış)
- [Tech Stack](#tech-stack)
- [Ekip Yapısı ve Sorumluluklar](#ekip-yapısı-ve-sorumluluklar)
- [Repo Yapısı](#repo-yapısı)
- [Kurulum](#kurulum)
- [Geliştirme Kuralları](#geliştirme-kuralları)

---

## Proje Özeti

Radyo teleskop dizilerinden (EHT vb.) elde edilen düşük çözünürlüklü ve bulanık kara delik görüntülerinin, derin öğrenme tabanlı süper-rezolüsyon ve gürültü giderimi teknikleriyle netleştirilmesi. Proje, model geliştirmenin yanı sıra uçtan uca bir MLOps altyapısı, veri pipeline'ı, Go tabanlı API gateway ve React frontend arayüzü kurmayı kapsar.

**Ekip:** 5 Stajyer  
**Süre:** 12 Hafta  
**GPU:** 1x NVIDIA L40S (48 GB VRAM)

---

## Problem Tanımı

Kara delik görüntüleri birden fazla fiziksel ve enstrümantal nedenden dolayı bozuk ve bulanık elde edilir:

### Kırınım Limiti (Diffraction Limit)

Bir teleskopun açısal çözünürlüğü θ ≈ λ/D formülüyle belirlenir. EHT, 1.3 mm dalga boyunda (230 GHz) gözlem yapar. Dünya çapında bir baz hattı (~10.700 km) ile bile açısal çözünürlük ~20 mikro-arcsaniye (μas) düzeyinde kalır; bu da kara deliğin olay ufku ölçeğinde yalnızca birkaç piksellik bir görüntüye karşılık gelir.

### Seyrek UV-Düzlemi Örneklemesi (Sparse UV-Coverage)

VLBI (Very Long Baseline Interferometry) tekniğinde her teleskop çifti, Fourier uzayında (UV-düzlemi) yalnızca tek bir noktayı örnekler. Yer yüzeyindeki sınırlı teleskop sayısı nedeniyle UV-düzleminin büyük bölümü boş kalır. Van Cittert-Zernike teoremine göre görüntü, bu visibilite verilerinin ters Fourier dönüşümüyle elde edilir; eksik frekans bilgisi doğrudan görüntüde artifact ve belirsizlik yaratır.

### Point Spread Function (PSF) / Dirty Beam

Eksik UV-coverage'ın doğal sonucu olarak, interferometrik dizinin PSF'i (dirty beam) ideal bir Airy diskinden çok uzaktır. Gözlenen görüntü, gerçek gökyüzü parlaklık dağılımının bu düzensiz PSF ile konvolüsyonudur:

```
I_observed(x,y) = I_true(x,y) * PSF(x,y) + noise
```

Bu konvolüsyon yüksek frekanslı detayları bastırarak bulanıklığa yol açar.

### Termal Gürültü ve Sistem Sıcaklığı (T_sys)

Her alıcının sistem sıcaklığı (T_sys), termal gürültünün alt sınırını belirler. Sinyal-gürültü oranı:

```
SNR ∝ S · √(Δν · τ) / T_sys
```

ile ifade edilir (S: kaynak akısı, Δν: bant genişliği, τ: integrasyon süresi). Milimetre dalga boylarında atmosferik su buharı emilimi T_sys'i yükselterek SNR'yi ciddi şekilde düşürür.

### Atmosferik Faz Bozulmaları (Tropospheric Phase Corruption)

Milimetre dalga boylarında troposferdeki türbülanslı su buharı, gelen sinyalin fazını rastgele bozar. Bu faz hataları visibilite verilerinde koherans kaybına neden olur ve kalibre edilmediğinde görüntüde sahte yapılar (spurious structures) oluşturur.

### Baz Hattı Kalibrasyonu (Baseline Calibration Errors)

Her teleskop çifti arasındaki kazanç (gain) farkları, saat senkronizasyon hataları ve polarizasyon sızıntıları (polarization leakage), visibilite genliklerinde ve fazlarında sistematik hatalara yol açar. Bu hatalar, CLEAN veya MEM gibi klasik görüntü rekonstrüksiyon algoritmalarının çıktısını doğrudan etkiler.

### Amaç

Bulanık ve gürültülü girdi görüntüsünden → fiziksel olarak tutarlı, yüksek çözünürlüklü kara delik görüntüsü üretmek.

---

## Mimari Genel Bakış

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐  │
│   │  FITS /  │───▶│  Data        │───▶│  Feature     │───▶│  Training   │  │
│   │  HDF5    │    │  Pipeline    │    │  Store       │    │  Pipeline   │  │
│   │  (Raw)   │    │  (Airflow)   │    │  (MinIO+DVC) │    │  (PyTorch)  │  │
│   └──────────┘    └──────────────┘    └──────────────┘    └──────┬──────┘  │
│                                                                  │         │
│                                                                  ▼         │
│                                                           ┌─────────────┐  │
│                                                           │   MLflow    │  │
│                                                           │  (Registry) │  │
│                                                           └──────┬──────┘  │
│                                                                  │         │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐           │         │
│   │  React   │◀──▶│  Go API      │◀──▶│  Python      │◀──────────┘         │
│   │  Frontend│    │  Gateway     │    │  Inference   │                     │
│   │          │    │  (REST)      │    │  (gRPC)      │                     │
│   └──────────┘    └──────────────┘    └──────────────┘                     │
│                          │                                                  │
│                          ▼                                                  │
│                   ┌──────────────┐    ┌──────────────┐                     │
│                   │  Prometheus  │───▶│   Grafana    │                     │
│                   │  (Metrics)   │    │  (Dashboard) │                     │
│                   └──────────────┘    └──────────────┘                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Akış:**

1. Ham teleskop verileri (FITS/HDF5) Airflow DAG'ları ile ingest edilir ve işlenir.
2. İşlenmiş veriler DVC ile versiyonlanarak MinIO'ya yazılır.
3. PyTorch ile model eğitimi yapılır, tüm deneyler MLflow'a loglanır.
4. En iyi model MLflow Registry üzerinden promote edilir.
5. Python gRPC servisi modeli yükler ve inference yapar.
6. Go API Gateway, dış dünyaya REST API sunar; inference isteklerini gRPC ile Python servisine iletir.
7. React frontend, Go API üzerinden kullanıcıya görüntü yükleme ve sonuç görüntüleme arayüzü sağlar.
8. Prometheus tüm servislerin metriklerini toplar, Grafana ile görselleştirilir.

---

## Tech Stack

### Veri (Data Engineering)

| Katman | Teknoloji | Açıklama |
|--------|-----------|----------|
| Veri İşleme | NumPy, SciPy, OpenCV, scikit-image | Görüntü manipülasyonu, sinyal işleme |
| Astronomi | astropy, eht-imaging | FITS dosya okuma, VLBI veri işleme, simülasyon |
| Veri Versiyonlama | DVC | Veri setlerinin Git-benzeri versiyonlanması |
| Veri Kalitesi | Great Expectations | Otomatik veri doğrulama ve profiling |
| Object Storage | MinIO | S3-uyumlu lokal depolama |

### ML (Machine Learning)

| Katman | Teknoloji | Açıklama |
|--------|-----------|----------|
| Dil | Python 3.11+ | Ana geliştirme dili |
| Derin Öğrenme | PyTorch 2.x | Model geliştirme ve eğitim |
| Deney Takibi | MLflow | Experiment tracking, model registry, artifact store |
| Hiperparametre | Optuna | Otomatik hiperparametre optimizasyonu |
| Inference | gRPC + protobuf | Model servis iletişim protokolü |

### Frontend

| Katman | Teknoloji | Açıklama |
|--------|-----------|----------|
| Framework | React 18+ (TypeScript) | SPA frontend uygulaması |
| Styling | Tailwind CSS | Utility-first CSS framework |
| State | Zustand veya React Query | Sunucu state yönetimi ve caching |
| Görselleştirme | Three.js veya D3.js | Kara delik görüntülerinin interaktif görselleştirmesi |

### API Gateway

| Katman | Teknoloji | Açıklama |
|--------|-----------|----------|
| Dil | Go 1.22+ | API gateway geliştirme dili |
| HTTP Router | Gin veya Echo | Yüksek performanslı HTTP framework |
| gRPC Client | google.golang.org/grpc | Python inference servisine bağlantı |
| Validation | go-playground/validator | Request validation |
| Docs | Swagger / OpenAPI 3.0 | Otomatik API dokümantasyonu |

### MLOps

| Katman | Teknoloji | Açıklama |
|--------|-----------|----------|
| Orkestrasyon | Apache Airflow | DAG tabanlı pipeline yönetimi |
| Konteyner | Docker, Docker Compose | Servis izolasyonu ve ortam tutarlılığı |
| CI/CD | GitHub Actions | Otomatik test, build, deploy |
| Versiyon Kontrol | Git + GitHub | Kod versiyonlama ve code review |

### Monitoring

| Katman | Teknoloji | Açıklama |
|--------|-----------|----------|
| Metrikler | Prometheus | Zaman serisi metrik toplama |
| Dashboard | Grafana | Metrik görselleştirme ve alerting |
| Model Drift | Evidently AI | Data drift ve model performance monitoring |

---

## Ekip Yapısı ve Sorumluluklar

### Stajyer 1 — Data Engineer

Veri pipeline'ının sahibi. FITS/HDF5 dosyalarının parse edilmesinden sentetik veri üretimine, DVC versiyonlamadan Great Expectations validation suite'ine kadar tüm veri akışından sorumlu.

**Araştırması gereken konular:**
- FITS dosya formatı ve astropy ile okuma/yazma
- eht-imaging kütüphanesi ile GRMHD simülasyonlarından görüntü üretimi
- PSF modelleme ve sentetik degradation pipeline tasarımı
- Airflow DAG yazımı ve scheduling
- DVC remote storage konfigürasyonu (MinIO backend)
- Great Expectations ile veri profiling ve expectation suite oluşturma

---

### Stajyer 2 — ML Engineer (Model Geliştirme)

Model mimarisinin ve eğitim sürecinin sahibi. Baseline'dan SOTA modellere kadar tüm model geliştirme, eğitim ve hiperparametre optimizasyonundan sorumlu.

**Araştırması gereken konular:**
- Image super-resolution literatürü (SRCNN → EDSR → ESRGAN → Real-ESRGAN → Restormer)
- GAN eğitim dinamikleri (mode collapse, training instability) ve çözüm yöntemleri
- Physics-informed neural networks ve custom loss function tasarımı
- Progressive training stratejileri
- Mixed precision training (torch.amp) ve gradient accumulation
- Optuna ile hiperparametre arama stratejileri

---

### Stajyer 3 — ML Engineer (Değerlendirme & Optimizasyon)

Model kalitesinin ve inference performansının sahibi. Metrik implementasyonu, benchmark suite, model optimizasyonu (ONNX, TensorRT) ve gRPC inference servisinden sorumlu.

**Araştırması gereken konular:**
- Görüntü kalite metrikleri: PSNR, SSIM, LPIPS, FID — matematiksel temelleri ve implementasyonu
- Fiziksel tutarlılık metriği tasarımı (PSF consistency check)
- ONNX export ve TensorRT ile model optimizasyonu
- gRPC + protobuf ile Python inference servisi geliştirme
- Model profiling ve latency analizi (torch.profiler)
- MLflow model registry entegrasyonu ve artifact yönetimi

---

### Stajyer 4 — MLOps Engineer

Otomasyon ve altyapının sahibi. CI/CD pipeline'ları, Docker ortamları, Airflow kurulumu, MLflow konfigürasyonu ve deployment süreçlerinden sorumlu.

**Araştırması gereken konular:**
- Docker multi-stage build ve image optimizasyonu
- Docker Compose ile multi-service orkestrasyon
- GitHub Actions workflow tasarımı (matrix builds, caching, secrets)
- MLflow Tracking Server kurulumu (backend store + artifact store)
- Airflow kurulumu ve DAG best practices
- MinIO kurulumu ve S3-uyumlu bucket yönetimi
- Otomatik model validation ve staging → production promotion kuralları

---

### Stajyer 5 — Frontend & API Gateway Engineer

Kullanıcıya dokunan tüm katmanların sahibi. Go API gateway, React frontend, Prometheus/Grafana monitoring ve Evidently AI drift detection'dan sorumlu.

**Araştırması gereken konular:**
- Go ile REST API geliştirme (Gin veya Echo framework)
- Go gRPC client implementasyonu ve connection pooling
- Protobuf schema tanımlama (.proto dosyaları)
- React + TypeScript ile SPA geliştirme
- Dosya upload/download handling (multipart form, streaming response)
- Prometheus client library (Go ve Python) ile custom metrik tanımlama
- Grafana dashboard provisioning (JSON model)
- Evidently AI ile data drift ve model performance raporu oluşturma

---

## Repo Yapısı

```
deephorizon/
├── README.md
├── docker-compose.yml
├── Makefile
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── train.yml
│       └── deploy.yml
│
├── data-pipeline/                  # Stajyer 1
│   ├── dags/                       # Airflow DAG'ları
│   ├── src/
│   │   ├── fits_parser.py
│   │   ├── synthetic_generator.py
│   │   ├── psf_model.py
│   │   ├── augmentation.py
│   │   └── validation/             # Great Expectations
│   ├── dvc.yaml
│   └── requirements.txt
│
├── ml/                             # Stajyer 2 + 3
│   ├── src/
│   │   ├── models/
│   │   │   ├── unet.py
│   │   │   ├── pix2pix.py
│   │   │   ├── esrgan.py
│   │   │   └── restormer.py
│   │   ├── training/
│   │   │   ├── train.py
│   │   │   ├── config.py
│   │   │   └── losses.py
│   │   ├── evaluation/
│   │   │   ├── metrics.py
│   │   │   ├── physics_check.py
│   │   │   └── benchmark.py
│   │   └── serving/
│   │       ├── inference_server.py  # gRPC server
│   │       └── proto/
│   │           └── inference.proto
│   ├── notebooks/
│   ├── configs/                     # Hydra/YAML config dosyaları
│   └── requirements.txt
│
├── api/                             # Stajyer 5 (Go)
│   ├── cmd/
│   │   └── server/
│   │       └── main.go
│   ├── internal/
│   │   ├── handler/                 # HTTP handlers
│   │   ├── middleware/              # Auth, CORS, logging
│   │   ├── grpc/                    # gRPC client (Python inference'a bağlantı)
│   │   └── model/                   # Request/response structs
│   ├── proto/
│   │   └── inference.proto          # Paylaşılan proto tanımı
│   ├── go.mod
│   └── go.sum
│
├── frontend/                        # Stajyer 5
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── services/                # API client
│   │   └── App.tsx
│   ├── package.json
│   └── tsconfig.json
│
├── infra/                           # Stajyer 4
│   ├── docker/
│   │   ├── Dockerfile.data-pipeline
│   │   ├── Dockerfile.ml-train
│   │   ├── Dockerfile.ml-serve
│   │   ├── Dockerfile.api
│   │   └── Dockerfile.frontend
│   ├── prometheus/
│   │   └── prometheus.yml
│   ├── grafana/
│   │   └── dashboards/
│   └── airflow/
│       └── airflow.cfg
│
├── docs/
│   ├── data_card.md
│   ├── model_card.md
│   ├── api_guide.md
│   └── runbook.md
│
└── tests/
    ├── data-pipeline/
    ├── ml/
    └── api/
```

---

## Kurulum

### Gereksinimler

- Docker & Docker Compose
- NVIDIA Driver + CUDA Toolkit (GPU eğitim için)
- Python 3.11+
- Go 1.22+
- Node.js 20+ (frontend için)
- Git

### Hızlı Başlangıç

```bash
# Repo'yu klonla
git clone https://github.com/<org>/deephorizon.git
cd deephorizon

# Tüm servisleri ayağa kaldır
docker compose up -d

# Servislere eriş
# Frontend:    http://localhost:3000
# Go API:      http://localhost:8080
# MLflow:      http://localhost:5000
# Airflow:     http://localhost:8081
# Grafana:     http://localhost:3001
# MinIO:       http://localhost:9001
# Prometheus:  http://localhost:9090
```

### Servis Bazlı Geliştirme

```bash
# Data Pipeline
cd data-pipeline
pip install -r requirements.txt

# ML
cd ml
pip install -r requirements.txt

# API (Go)
cd api
go mod download
go run cmd/server/main.go

# Frontend
cd frontend
npm install
npm run dev
```

---

## Geliştirme Kuralları

### Git Workflow

- **Ana branch:** `main` (protected, merge sadece PR ile)
- **Branch isimlendirme:** `feature/<stajyer-adı>/<kısa-açıklama>` (örn: `feature/ahmet/fits-parser`)
- Her PR en az 1 review almalı
- PR açıklamasında ne yapıldığı ve nasıl test edildiği yazılmalı

### Commit Convention

```
<type>(<scope>): <açıklama>

type:  feat | fix | refactor | docs | test | ci | chore
scope: data | ml | api | frontend | infra | docs
```

Örnekler:
```
feat(data): add FITS parser with astropy
fix(ml): resolve CUDA OOM in ESRGAN training
feat(api): implement /enhance endpoint with gRPC client
docs(ml): add model card for pix2pix v1
ci(infra): add Docker build caching to GitHub Actions
```

### Code Review Kuralları

- Kendi PR'ını kendin merge edemezsin
- Review'da en az şu kontrol edilmeli: çalışıyor mu, test var mı, dokümantasyon güncellendi mi
- Review 24 saat içinde yapılmalı

### Dokümantasyon

- Her modül kendi `README.md` dosyasına sahip olmalı
- Public fonksiyonlar docstring ile belgelenmeli
- API endpoint'leri Swagger/OpenAPI ile otomatik dokümante edilmeli
- Önemli mimari kararlar `docs/` altında ADR (Architecture Decision Record) formatında tutulmalı