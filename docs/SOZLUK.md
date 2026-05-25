# 📖 Teknik Sözlük

DeepHorizon repo'sunda karşılaşacağın terimlerin hızlı açıklamaları. "Bu ne ya" dedirten her şey burada — alfabetik değil, **kullandığımız yerlere göre gruplanmış**. Bir terimin başka bir terime referansı varsa `→ Bkz.` ile işaretli.

---

## Repo / Git

### `.gitkeep`
Boş klasörü Git'e kabul ettiren konvansiyonel bir dosya adı (sıfır byte). Git boş klasörleri takip etmez; içine `.gitkeep` koyarsan klasör commit olur. Standart bir Git özelliği **değil**, sadece bir gelenek. Klasörde gerçek dosya olunca silinir.

### `.gitignore`
Git'in **takip etmemesi** gereken dosya/klasör desenleri. Bizde `.venv`, `node_modules`, `__pycache__`, `.env*` gibi şeyler ignore'lu.

### `.gitattributes`
Dosya türü başına özel kurallar (line endings, diff stratejisi, LFS). Henüz bizde yok.

### `app-of-apps` (Argo CD pattern)
Tek bir kök Argo CD `Application`'ı bütün diğer Application'ları izler. Yeni servis = yeni manifest, `argocd app create` komutu çalıştırmaya gerek yok. Bizim `infra/k8s/app-of-apps.yaml` budur.

---

## Diller arası kontrat — Protocol Buffers / gRPC

### `proto` / `.proto` dosyası
**Protocol Buffers**'ın kısa adı. Servisler ve mesajları tanımlayan bir IDL (interface definition language). Bizde `proto/deephorizon/v1/inference.proto` Go ile Python'un üzerinde anlaştığı sözleşme.

### `protobuf`
Aynı şeyin uzun adı. "Proto" denildiğinde bu kastedilir.

### `gRPC`
Google'ın RPC framework'ü. HTTP/2 üzerinde binary protobuf mesajlarıyla çalışır, REST'ten hızlı. Bizim Go gateway → Python inference servisi iletişimi gRPC.

### gRPC vs REST farkı
| | gRPC | REST |
|:---|:---|:---|
| Veri formatı | Binary (protobuf) | JSON |
| Hız | ~5x daha hızlı | Yavaş ama okunabilir |
| Tarayıcıdan kullanım | Zor (gRPC-Web gerek) | Native |
| Tipler | Compile-time, garantili | Runtime, JSON validation |

Bu yüzden frontend Go'ya REST atıyor, Go Python'a gRPC atıyor.

### `buf`
Protobuf'lar için lint + breaking-change tespit aracı. `buf lint` proto stilini kontrol eder, `buf generate` Go ve Python kod stub'larını üretir. Bizde `proto/buf.yaml` ile yapılandırıldı.

### Stub (kod stub'ı)
`buf generate` çalıştırınca .proto'dan otomatik üretilen Go/Python kodu. Bunları **elle düzenleme**. `.proto`'yu değiştir, yeniden generate et.

---

## Container & Kubernetes

### `Pod`
Kubernetes'in en küçük çalışan birimi. 1+ container çalıştırır. Genelde 1 pod = 1 container düşün.

### `Deployment`
Bir pod tipinin **kaç tane çalışacağını** ve nasıl güncelleneceğini tanımlar. Bizim Go API gateway `replicas: 2` ile Deployment olarak yaşar.

### `Job`
Tek seferlik veya zamanlı koşan pod. Bizim model eğitimi bir Job'tır — bitince ölür, durmaz.

### `Service`
Pod'lara stabil bir DNS adı verir. Pod'lar IP değiştirir, Service değişmez. Pod gruplarına trafik dağıtır.

### `Ingress`
Cluster dışından gelen HTTP/HTTPS isteklerini doğru Service'e yönlendiren bileşen — **klasik kurulumda**. **Bizde farklı:** sunucu network yapısı nedeniyle cluster-içi Ingress controller (örn. ingress-nginx) **kullanmıyoruz**. Bunun yerine `services/api` ve `services/frontend` Service'leri `NodePort` olarak açılır; sunucunun host'unda çalışan **NGINX Proxy Manager** (cluster dışı, host'ta Docker container) bu NodePort'lara reverse proxy yapar, TLS sertifikalarını yönetir.

```
Internet → host:443 (NGINX Proxy Manager) → MicroK8s NodePort → Service → Pod
```

Bu yüzden bizim `infra/k8s/`'de Ingress manifest'i **yok**, sadece NodePort Service var. SSL/domain yönetimi NPM panelinden yapılır, Git'te değil.

### `NGINX Proxy Manager` (NPM)
Cluster **dışında** host üzerinde çalışan GUI tabanlı reverse proxy + Let's Encrypt yöneticisi (Docker container). Bizim "Ingress controller yerine geçen şey". Yapılandırması NPM'in web arayüzünden yapılır, repo'da YAML olarak tutulmaz.

> ⚠️ Bu, GitOps'tan kopuk tek bileşen. Cluster içine **hiç manifest yazma** — NPM'in kendi sertifika/host konfigürasyonu admin panelinde yaşıyor. Domain veya SSL değişikliğini Argo CD değil, NPM yöneticisi yapar.

### `NodePort` (Service tipi)
Service'i her node'un belirli bir portunda (örn. `30080`) açan tip. Cluster dışından erişilebilir hâle getirir. Bizim API ve frontend Service'leri NodePort, NGINX Proxy Manager bu portlara yönleniyor.

### `Namespace`
Cluster içindeki "klasör" — kaynakları gruplayıp izole eder. Bizde 4 tane: `deephorizon-{data,ml,app,monitor}`.

### `kustomize`
Aynı manifest'i farklı ortamlarda (dev/staging/prod) farklı parametrelerle render eden tool. Base + overlay deseni. `infra/k8s/*/kustomization.yaml` ile yaşar.

### `Helm`
Kubernetes paket yöneticisi (npm gibi). `helm install argo-cd argo/argo-cd` der, üçüncü parti chart kurarsın. Biz Sealed Secrets ve gpu-operator için Helm kullanıyoruz, kendi servislerimiz için kustomize.

### `Helm chart`
Helm'in paketi — şablonlanmış manifest'ler + varsayılan değerler.

### `MicroK8s`
Hafif, tek-binary Kubernetes dağıtımı. Snap ile kurulur, GPU addon'u var. Tek node cluster için ideal.

### `Argo CD`
GitOps tool'u — Git'teki manifest değişikliklerini cluster'a otomatik uygular. Kuralı: cluster'a kimse elle `kubectl apply` yapmaz, sadece Git'e push edilir.

### `GitOps`
"Git deposu = doğruluk kaynağı" prensibi. Cluster'da ne çalıştığını görmek için → Git'e bak.

### `Sealed Secrets`
Bitnami'nin Kubernetes Secret'larını şifreleyip Git'e güvenle commit etmene izin veren controller'ı. Cluster içindeki private key ile çözer.

### `kubeseal`
Sealed Secrets'ın CLI'ı. `kubectl create secret ... | kubeseal -o yaml > sealed.yaml` ile düz Secret'ı şifrelenmiş `SealedSecret`'a çevirir.

### `nodeSelector`
Pod'u sadece belirli etiketteki node'lara yerleştir. Bizim eğitim Job'ı `workload: training` etiketli node istiyor.

### `PriorityClass`
Pod öncelik sınıfı — kaynak baskısında düşük öncelikli pod'lar tahliye edilir. Eğitim `low-priority`, inference `high-priority`.

### `NVIDIA MIG` (Multi-Instance GPU)
Tek bir GPU'yu (L40S, A100, H100 gibi) donanım düzeyinde küçük "GPU dilimlerine" bölme. Bizde planlı ama 11. haftaya kadar kapalı — kurulum maliyeti şu an için fazla.

### `gpu-operator`
NVIDIA'nın Kubernetes operator'ü — driver, container runtime, device plugin, MIG yönetimi otomatik.

---

## CI / CD

### `CI` (Continuous Integration)
Her PR/push'ta otomatik lint, test, build çalıştırma. Bizde GitHub Actions.

### `CD` (Continuous Deployment)
Build edilen artefakt'ı otomatik deploy etme. Bizde Argo CD.

### `workflow`
GitHub Actions'ın bir CI tanımı (`.github/workflows/*.yml`).

### `breaking change` (proto bağlamında)
Eski client'ları kıran proto değişikliği — alan silmek, ID değiştirmek, tip değiştirmek. `buf breaking` PR'larda kontrol eder.

---

## Python / Bağımlılık yönetimi

### `uv`
Astral'in yeni Python paket yöneticisi. `pip + venv + pip-tools`'u tek binary'de birleştirir, çok hızlı (Rust ile yazıldı). Bizde resmi tool.

### `pyproject.toml`
Modern Python projeleri için tek konfig dosyası. PEP 518/621. `setup.py` ve `requirements.txt`'in halefi.

### `extras` (optional-dependencies)
`pip install -e ".[ml]"` ile sadece ML bağımlılıklarını yüklersin. Bizde `data`, `ml`, `serving`, `dev` extras'ı var.

### `venv` / virtualenv
İzole bir Python ortamı. Sistem Python'una dokunmadan paket kurarsın. `uv` venv yönetimini de yapar.

### `pkg_resources`
setuptools'un eski API'si. ehtim hâlâ kullanıyor, setuptools 82+ kaldıracak. Bizde `setuptools<82` pin'i var.

### `__init__.py`
Bir klasörü Python paketi haline getiren dosya. İçi boş bile olabilir.

---

## Veri / ML / Astronomi

### `FITS` (Flexible Image Transport System)
Astronomide standart görüntü formatı. astropy ile okunur.

### `UVFITS`
EHT'nin görünürlük (visibility) verisini sakladığı FITS varyantı. Ham gözlem verisi.

### `HDF5`
Hierarchical Data Format 5 — büyük sayısal veri için binary format. ML'de yaygın.

### `VLBI` (Very Long Baseline Interferometry)
Birbirinden çok uzak teleskopları sanal tek bir teleskop gibi kullanma tekniği. EHT bunu yapar.

### `UV-plane`
VLBI'da her teleskop çiftinin örneklediği Fourier uzayı noktası. Bütün noktalar bir araya gelirse görüntü oluşur.

### `PSF` (Point Spread Function)
Teleskobun bir nokta kaynağı nasıl bulanıklaştırdığı. Bizim için "dirty beam" denilen şey.

### `eht-imaging` / `ehtim`
EHT collaboration'ın görüntüleme kütüphanesi. `pip install ehtim`. Bizde sentetik veri üretimi için kullanılıyor.

### `DVC` (Data Version Control)
Git benzeri ama büyük dosyalar için. Veri setlerini versiyonlar, asıl içerik MinIO/S3'te durur, Git sadece pointer tutar.

### `Great Expectations`
Veri doğrulama framework'ü. "Bu sütun null olmamalı, şu değer aralığında olmalı" gibi kontrolleri çalıştırır.

### `MinIO`
S3 uyumlu, self-hosted nesne depolama. Cluster içinde çalışır, AWS S3 maliyeti yok.

### `MLflow`
Deney takip + model registry. Her eğitim koşusunu (hiperparametre, metrik, artefakt) loglar.

### `MLflow Registry`
Eğitilen modelleri **Staging → Production** olarak terfi ettirdiğin yer. Doğrulama kapısı geçmeden production'a çıkmaz.

### `Optuna`
Hiperparametre optimizasyon kütüphanesi. **TPE** (Tree-structured Parzen Estimator) ile akıllı arama yapar.

### `Hydra`
Yapılandırma yönetimi — eğitim konfiglerini YAML'lardan composable şekilde yöneten Facebook kütüphanesi.

### `PSNR` / `SSIM` / `LPIPS` / `FID`
Görüntü kalite metrikleri. PSNR/SSIM klasik (piksel/yapısal), LPIPS/FID derin öğrenme tabanlı (algısal). README'de hedefler var.

### `ONNX` (Open Neural Network Exchange)
Çerçeve-bağımsız model formatı. PyTorch'tan export → ONNX Runtime ile çalıştır. CPU/GPU/edge için optimize edilebilir.

### `TensorRT`
NVIDIA'nın inference optimizer'ı. ONNX'i GPU için maksimum hıza derler.

### `torch.amp` (Automatic Mixed Precision)
FP16/FP32 karışık kullanarak eğitimi hızlandıran PyTorch özelliği. Bellek de yarıya iner.

### `gradient accumulation`
Küçük batch'leri toplayıp büyük batch gibi davranma. GPU belleği yetmediğinde işe yarar.

### `bikübik upsample`
Klasik (ML olmayan) görüntü büyütme algoritması. Bizim baseline (PSNR ~18 dB) bu.

### `GRMHD`
General Relativistic Magnetohydrodynamics. Kara delik etrafındaki plazma simülasyonu. Eğitim verimiz bu modellerden geliyor.

---

## Frontend

### Next.js `App Router`
Next.js 13+'ın yeni router'ı. `app/` klasör yapısı, file-based routing, Server Components default. Eski `pages/` router'ından farklı.

### `Server Component` vs `Client Component`
Next.js'de varsayılan tüm bileşenler sunucuda render edilir (Server Component). `"use client"` direktifi koyunca tarayıcıda interaktif olur. Bizim çoğu sayfa Client (form, state).

### `output: "standalone"` (Next.js)
Build çıktısını minimum dependency ile paketler. Production Docker imajı 400 MB yerine ~80 MB olur.

### `Zustand`
Hafif state management kütüphanesi. Redux'tan çok daha az boilerplate. Bizim global state'imiz.

### `TanStack Query` (eski adı React Query)
Server state önbellekleme, refetch, optimistic update kütüphanesi. API çağrılarını yönetir.

### `Three.js`
WebGL üzerinde 3B grafik. Bizde kara delik görüntü viewer'ı için.

### `D3.js`
Data-driven document — interaktif grafik/chart. PSNR/SSIM zaman grafikleri için.

### `Tailwind CSS`
Utility-first CSS. `<div class="flex gap-4 p-2 bg-zinc-800">` gibi. Komponent stillemek için CSS yazmıyorsun, class compose ediyorsun. **v3 kullanıyoruz**, v4 plugin churn'ünden kaçındık.

---

## Gözlemlenebilirlik

### `Prometheus`
Metrik toplama sistemi. Her servis `/metrics` endpoint'i sunar, Prometheus periyodik olarak scrape eder.

### `Grafana`
Prometheus'tan gelen metrikleri dashboard olarak gösterir.

### `Evidently AI`
Veri kayması (data drift) ve model performans bozulması tespit kütüphanesi. Production'da modelin tahminlerini canlı izler.

### `metric`
Prometheus'un topladığı sayı (counter, gauge, histogram). Örnek: `enhance_requests_total`.

### `OpenTelemetry`
Telemetri standardı (metrik + log + trace). Şu an roadmap'te değil.

---

## Yapay Zeka Mimarileri

### `U-Net`
Encoder-decoder + skip connections. Görüntü-görüntü çevirisi için golden standard.

### `Pix2Pix`
Koşullu GAN. U-Net generator + PatchGAN discriminator.

### `GAN` (Generative Adversarial Network)
Generator vs Discriminator yarışı. Sıkı eğitim, mode collapse riski yüksek.

### `ESRGAN`
Enhanced Super-Resolution GAN. RRDB blokları + relativistic discriminator. SR'da SOTA'ya yakın.

### `Restormer`
Transformer tabanlı görüntü restorasyon. Multi-Dconv head Transposed Attention (MDTA). Bizim **stretch goal**.

### `RRDB` (Residual-in-Residual Dense Block)
ESRGAN'ın generator yapı taşı.

### `attention mechanism`
Transformer'ın temel bileşeni. Her piksel/token diğerleriyle ilişkisini öğrenir.

---

## Diğer

### `ADR` (Architecture Decision Record)
Mimari kararları belgeleyen kısa dokümanlar. **Context · Decision · Consequences** formatında. `docs/adr/` klasöründe yaşar.

### `runbook`
Bir hata/incident için adım adım kurtarma talimatı. "Prometheus tarafında alert geldi → şunu yap" gibi.

### `SOTA` (State of the Art)
Mevcut en iyi sonuç. Restormer SR/denoising'de SOTA'ya yakın.

### `go/no-go gate`
Bir aşamaya geçmeden önce **devam edip etmeyeceğine** karar verdiğin nokta. 8. haftada SSIM ≥ 0.85 → Restormer'a gir, değilse mevcudu sağlamlaştır.

### `bus factor`
"X kişi otobüsün altında kalsa proje devam eder mi?" — düşükse risk. Biz MLOps'u tek stajyere bağlamayarak bus factor'u 3'e çıkardık.

### `pinning` (paket pin'lemek)
Bir paketin sürümünü sabit tutmak. `setuptools<82` gibi. README'de `pyproject.toml`'da örnekler var.

### `monorepo`
Birden fazla servisin (data + ml + api + frontend) tek bir Git deposunda yaşaması. Bizimki budur.

---

## Bilmediğin bir terim mi var?

PR aç, bu listeye ekle. **Sözlük büyüdükçe stajyer onboarding'i hızlanır.**
