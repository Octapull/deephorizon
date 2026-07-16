# Bugün Ne Yaptık? — DevOps Bootstrap Özeti (14 Temmuz 2026)

Bu doküman, DeepHorizon'un GPU sunucusunun **çıplak bir Ubuntu kurulumundan, üzerinde
veri barındıran GitOps'lu bir ML platformuna** dönüştürüldüğü günün özetidir. Her
adımda *ne yaptık, neden yaptık, nasıl yaptık* sorularına cevap verir. Komut komut
teknik günlük ve hata analizleri için: [`docs/DEVOPS.md`](docs/DEVOPS.md) ·
Veri düzeni ve erişim rehberi için: [`docs/DATA.md`](docs/DATA.md)

---

## Sabah elimizde ne vardı, akşam ne oldu?

| | Sabah | Akşam |
|:---|:---|:---|
| Sunucu | Çıplak Ubuntu 22.04, sürücüsüz L40S | GPU'lu, tek node'luk Kubernetes platformu |
| Deploy | — | `git push` = deploy (Argo CD, app-of-apps) |
| Secret'lar | — | Git'te şifreli (Sealed Secrets), anahtarı yedekli |
| Dış dünya | — | NPM 80/443, UFW ile kilitli, admin panelleri tünel arkasında |
| Veri | Sadece üretim scriptleri | MinIO'da 20 GiB'lık ilk eğitim seti, erişim kontrollü |

---

## 1. Sunucuyu GPU'lu bir Kubernetes node'una çevirdik

**Ne:** NVIDIA sürücüsü + MicroK8s + GPU Operator.

**Neden:** Her şeyin temeli. Eğitim de inference da `nvidia.com/gpu` isteyen pod'lar
olarak koşacak; bunun için kernel'den scheduler'a kadar zincirin tamamının çalışması
gerekiyor.

**Nasıl:** Önce sistem güncellemesi + reboot (sürücü modülü mevcut kernel'e derlenir —
sıra önemli). Sürücüde `nvidia-driver-580-server` seçtik: "en yeni dalın bir gerisi"
kuralı (595 çok taze, 535 gereksiz eski) ve `apt-mark hold` ile sabitleme (habersiz
sürücü güncellemesi haftalarca sürecek eğitimi ortasında kırabilir). MicroK8s'i sürümlü
kanaldan (1.32/stable) kurduk, GPU addon'ını host sürücü üstüne oturttuk. Kanıt:
`cuda-vector-add` test pod'u → `Test PASSED`.

## 2. Secret yönetimini kurduk (Sealed Secrets)

**Ne:** Sealed Secrets controller + `kubeseal` CLI + master key yedeği.

**Neden:** Proje kuralı net: kaynak kodda/`.env`'de secret yaşamaz, ama GitOps'ta her
şey Git'te olmalı. Çözüm: secret'lar **şifrelenmiş** commit edilir, yalnızca cluster
içindeki controller çözebilir.

**Nasıl:** Controller Helm'le kuruldu, `kubeseal` controller'la aynı sürümden indirildi,
uçtan uca test edildi (seal → apply → çözülen değeri geri oku). En kritik adım:
**controller'ın private key'ini yedekledik** — bu yedek olmadan cluster kaybında
Git'teki tüm SealedSecret'lar sonsuza dek çözülemez olurdu.

## 3. GitOps motorunu kurduk (Argo CD)

**Ne:** Argo CD + repo bağlantısı.

**Neden:** Hedef işleyiş "deploy = git push": kimse elle `kubectl apply` yapmaz,
cluster'ın gerçek durumu her zaman `main`'deki manifest'lerdir. Bunu izleyip uygulayan
motor Argo CD.

**Nasıl:** MicroK8s community addon'ıyla kurduk, UI'ı NodePort'a aldık, repo'yu
**declarative** bir Secret ile bağladık (repo public — credential bile gerekmedi).
Yol boyunca üç ayrı tuzağa düştük (localhost-NodePort meselesi, CLI/server sürüm
uyuşmazlığı, `--grpc-web` ihtiyacı) — hepsi çözümüyle `docs/DEVOPS.md`'de.

## 4. Dış kapıyı ve kilidi taktık (Docker + NPM + UFW)

**Ne:** Host'ta Docker, NGINX Proxy Manager, UFW firewall.

**Neden:** Proje bilinçli olarak cluster-içi Ingress kullanmıyor; TLS ve domain
yönlendirmesi host'taki NPM'de yaşayacak (`Internet → NPM:443 → NodePort → Pod`).
Firewall ile de dışarıya yalnızca 22/80/443 açık kalacak.

**Nasıl:** Docker resmi depodan kuruldu; NPM compose ile kalktı — admin paneli (81)
**bilinçli olarak sadece localhost'a bağlı**, erişim SSH tüneliyle (internete açık NPM
paneli klasik saldırı yüzeyidir). UFW açıldıktan sonra en sinsi sorunu yaşadık: pod'ların
API server'a erişimi sessizce kırıldı (DNS testi geçtiği halde!). Calico arayüzlerini
UFW'den muaf tutarak çözdük. Domain henüz yok; alınınca NPM'de proxy host'lar +
Let's Encrypt tanımlanacak, tüneller emekli olacak.

## 5. GitOps'u fiilen devreye aldık (app-of-apps + MinIO)

**Ne:** `infra/k8s/` manifest ağacı, root Application, ilk iş yükü olarak MinIO.

**Neden:** Kurulu bir Argo CD, ilk gerçek deploy'unu yapana kadar sadece bir vaattir.
MinIO'yu seçtik çünkü zincirin en altındaki bağımlılık: eğitim verisi, DVC remote'u ve
MLflow artifact'leri hep ona yaslanacak.

**Nasıl:** App-of-apps deseni: tek root Application, `infra/k8s/apps/` altındaki squad
Application'larını izliyor; onlar da kendi dizinlerini. MinIO pinli imajla, 100Gi
PVC'yle, credential'ları SealedSecret'tan okuyarak StatefulSet olarak tanımlandı.
PR → merge → `kubectl apply -f app-of-apps.yaml` (zincirin SON elle komutu) →
Argo CD her şeyi kendisi kurdu. Bundan sonra yeni servis eklemenin reçetesi:
*manifest yaz → Application ekle → secret'ı seal'le → PR aç.* (Sıradaki MLflow bu
reçeteyle, başka bir ekip üyesi tarafından kurulacak.)

## 6. Veri katmanını düzenledik ve ilk seti yükledik

**Ne:** Bucket düzeni, erişim kontrollü kullanıcılar, 20 GiB'lık eğitim seti.

**Neden:** ML ekibinin veriye "rahat ama kontrollü" erişmesi gerekiyor: herkes okuyabilsin
ama kimse yanlışlıkla seti bozamasın; parametreler değişince eski deneyler kırılmasın.

**Nasıl:** Üç bucket (`raw` / `datasets` / `mlflow`) + **versiyonlu prefix** sözleşmesi
(`training-512/v1/` — parametre değişirse `v2` açılır). `generate_training_data.py`
sunucuda koştu, çıktı `mc mirror` ile yüklendi: **10.000 çift, 20.000 obje, ~20 GiB**
(bu arada dokümanlardaki "~2.5 GB" tahmininin yanlış olduğunu ölçerek bulduk ve
düzelttik). Erişim için özel `ml-read` policy'si yazdık — MinIO'nun yerleşik
`readonly`'sinin listeleme içermediğini yaşayarak öğrendik. Üç yönlü doğrulama:
okuma ✓, yazma reddi ✓, kapsam dışı bucket reddi ✓.

---

## Günün ilkeleri (neden hep böyle yaptık?)

1. **Her şey Git'te, her şey izlenebilir** — manifest'ler, şifreli secret'lar,
   dokümanlar. Elle yapılan tek şey bootstrap'tı; o da günlüğe işlendi.
2. **Sürüm sabitle, `latest`'e güvenme** — sürücü hold'da, MicroK8s sürümlü kanalda,
   MinIO imajı pinli. Günün en uzun hata ayıklaması (`argocd` CLI) tam da bir
   `latest`'ten çıktı.
3. **En az yetki** — ML ekibi read-only ve iki bucket'la sınırlı; root credential
   kimseye verilmiyor; admin panelleri internete kapalı.
4. **Doğrulamadan ilerleme** — her adımın "beklenen çıktı"sı görülmeden sonrakine
   geçilmedi; yetersiz çıkan test de (UFW/DNS) dersiyle birlikte kaydedildi.
5. **Hatayı çöz, sonra dokümante et** — günün 8 hatası sebep-çözüm-ders formatında
   `docs/DEVOPS.md`'de; bir sonraki kişi aynı çukura düşmesin.

## Kalanlar (sahipleriyle)

| İş | Sahibi |
|:---|:---|
| MLflow kurulumu (`deephorizon-ml`, reçete hazır) | Ekipten başka üye |
| MinIO root parola rotasyonu | DevOps |
| Argo CD sürüm yükseltme (UI internete açılmadan önce) | DevOps |
| Domain + NPM proxy host'ları + Let's Encrypt | DevOps (domain bekleniyor) |
| Airflow DAG'ları (veri üretimi şimdilik elle) | Data squad |
