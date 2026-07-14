# DevOps Kurulum Günlüğü — GPU Sunucu Bootstrap

DeepHorizon'un GPU sunucusunun (Ubuntu 22.04 LTS, 1x NVIDIA L40S 48 GB) sıfırdan
**platform-hazır** hale getirilmesinin kaydı: neler kuruldu, hangi sırayla, hangi
hatalarla karşılaşıldı, neden oldu ve nasıl çözüldü. Kurulum **14 Temmuz 2026**'da
tamamlandı ve her adım sahada doğrulandı.

Zincir, [`infra/README.md`](../infra/README.md) → "Apply order" ile birebir aynı:

```
1) NVIDIA sürücü + MicroK8s + GPU addon
2) Sealed Secrets (controller + kubeseal)
3) Argo CD (+ repo bağlantısı)
4) Host Docker + NGINX Proxy Manager + UFW
```

Bu noktadan sonrası repo işidir: `infra/k8s/` altına manifest yazılır, `main`'e push
edilir, Argo CD deploy eder. Elle `kubectl apply` deploy akışının parçası değildir.

---

## 1. Ön Kontroller

| Kontrol | Komut | Sonuç |
|:---|:---|:---|
| OS / kernel | `cat /etc/os-release`, `uname -r` | Ubuntu 22.04.5, `5.15.0-185-generic` |
| GPU (PCI seviyesi) | `lspci \| grep -i nvidia` | L40S görünüyor |
| Secure Boot | `mokutil --sb-state` | `EFI variables are not supported` → bkz. Hata #1 |
| Güncelleme | `sudo apt update && sudo apt upgrade -y` + reboot | Kernel güncellemesi sürücüden **önce** yapıldı (sürücü modülü mevcut kernel'e derlenir; sonra güncellenirse kırılır) |

## 2. NVIDIA Sürücü (Host Driver)

**Karar:** Sürücü host'a kuruldu; MicroK8s GPU addon'unun (GPU Operator) konteyner-içi
sürücü seçeneği kullanılmadı. Gerekçe: operator sürücüsü her kernel güncellemesinde
yeniden derlenir ve tek-node kümede hata ayıklaması zordur; host sürücü daha öngörülebilir.

```bash
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers list --gpgpu          # dallar: 535 / 580 / 595 (+ -open varyantları)
sudo apt install -y nvidia-driver-580-server
sudo reboot
nvidia-smi                                # L40S, ~46 GB, Driver 580.xx ✓
sudo apt-mark hold nvidia-driver-580-server
```

- **Sürüm seçim kuralı:** listedeki en yeni `-server` dalının bir gerisi (595 dururken
  580) — en yeni dal en az test edilmiş olandır. L40S için minimum 535.
- **`-open` varyantları** (açık kaynak kernel modülü) kullanılmadı; GPU Operator
  zinciriyle en çok test edilmiş yol klasik `-server` paketi.
- **`apt-mark hold`:** unattended-upgrades'in sürücüyü habersiz güncelleyip haftalarca
  sürecek eğitim koşularını kırmaması için sürüm sabitlendi.

## 3. MicroK8s

```bash
sudo snap install microk8s --classic --channel=1.32/stable
sudo usermod -aG microk8s $USER && newgrp microk8s
microk8s status --wait-ready
microk8s enable hostpath-storage          # dns ve helm3 bu sürümde varsayılan geliyor
echo "alias kubectl='microk8s kubectl'" >> ~/.bashrc
microk8s config > ~/.kube/config && chmod 600 ~/.kube/config
```

- Kanal olarak `latest/stable` değil **sürümlü kanal** (1.32/stable) seçildi — habersiz
  minor atlaması GitOps kurulumunu kırabilir. Kurulan: v1.32.13.
- `~/.kube/config` cluster'a tam erişim verir: sunucudan dışarı kopyalanmaz, repoya girmez.
- `ingress` addon'u **bilinçli olarak kapalı** — TLS/yönlendirme host'taki NPM'de
  (proje kararı: "no cluster Ingress"), ayrıca 80/443'ü NPM'le çakıştırır.

## 4. GPU Addon (NVIDIA GPU Operator)

```bash
microk8s enable nvidia                    # host sürücüyü otomatik algıladı
kubectl get pods -n gpu-operator-resources   # validator'lar Running/Completed
kubectl get node -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}'   # → 1
```

Uçtan uca doğrulama: `nvidia.com/gpu: 1` isteyen `cuda-vector-add` test pod'u →
loglarda **`Test PASSED`**. Scheduler → device plugin → container toolkit → sürücü
zincirinin tamamı çalışıyor. (`nvidia-cuda-validator` pod'unun `Completed` durumda
kalması normaldir — tek seferlik doğrulama job'ıdır.)

## 5. Sealed Secrets

Amaç: secret'lar Git'e **şifrelenmiş** (`SealedSecret`) commit edilir; cluster'daki
controller bunları normal `Secret`'a çözer. Kaynak kodda/`.env`'de secret yaşamaz.

```bash
microk8s helm3 repo add sealed-secrets https://bitnami.github.io/sealed-secrets   # bkz. Hata #2
microk8s helm3 repo update
microk8s helm3 install sealed-secrets sealed-secrets/sealed-secrets \
  --namespace kube-system \
  --set fullnameOverride=sealed-secrets-controller
```

- `fullnameOverride=sealed-secrets-controller`: `kubeseal` CLI'ın varsayılan aradığı ad;
  aksi halde her çağrıya `--controller-name` bayrağı gerekir.
- **kubeseal CLI** controller ile aynı sürümden (0.38.4) GitHub release'inden kuruldu
  (`github.com/bitnami/sealed-secrets`).
- **Uçtan uca test:** `kubectl create secret --dry-run` → `kubeseal` → apply →
  controller'ın çözdüğü Secret'tan değer geri okundu (`world`) ✓
- **🔴 Master key yedeği alındı:** controller'ın private key'i sadece cluster'da yaşar;
  cluster kaybında yedek yoksa Git'teki tüm SealedSecret'lar kalıcı çözülemez olur.
  ```bash
  kubectl get secret -n kube-system -l sealedsecrets.bitnami.com/sealed-secrets-key \
    -o yaml > sealed-secrets-master-key.yaml
  # sunucudan alındı → parola kasasına → sunucuda shred -u ile silindi
  ```
  Geri yükleme: YAML'ı yeni cluster'da `kube-system`'e apply et, controller pod'unu
  yeniden başlat.
- **Seal'lerken dikkat:** şifreleme namespace **adına** scope'ludur — secret hangi
  namespace'te kullanılacaksa `-n <namespace>` ile seal'lenir (namespace'in henüz var
  olması gerekmez). Proje secret envanteri root README → "Secret Inventory" tablosunda.

## 6. Argo CD

```bash
microk8s enable community && microk8s enable argocd   # argocd namespace'ine kurar
kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath='{.data.password}' | base64 -d; echo    # ilk admin parolası
```

- Addon'un kurduğu server Service'i zaten NodePort'luydu: `443:30443`.
- UI: `https://<node-ip>:30443` (self-signed). Admin parolası değiştirildi, ardından
  `argocd-initial-admin-secret` silindi.
- **CLI girişi** (üç hatadan sonra — bkz. Hata #3/#4/#5) çalışan biçim:
  ```bash
  # CLI sürümü SERVER ile aynı olmalı (v2.7.2), latest DEĞİL:
  curl -fsSL -o argocd https://github.com/argoproj/argo-cd/releases/download/v2.7.2/argocd-linux-amd64
  sudo install -m 755 argocd /usr/local/bin/argocd
  argocd login <node-ip>:30443 --username admin --insecure --grpc-web
  ```
- **Repo bağlantısı** CLI'a bağımlı olmayan **declarative** yolla yapıldı (repo public
  olduğundan credential gerekmedi — bkz. Hata #6):
  ```bash
  kubectl apply -f - <<'EOF'
  apiVersion: v1
  kind: Secret
  metadata:
    name: repo-deephorizon
    namespace: argocd
    labels:
      argocd.argoproj.io/secret-type: repository
  stringData:
    type: git
    url: https://github.com/Octapull/deephorizon.git
  EOF
  ```
  Sonuç: Settings → Repositories → `CONNECTION STATUS: Successful` ✓
- **Bekleyen tek adım (manifest'ler yazılınca, tek seferlik):**
  `kubectl apply -f infra/k8s/app-of-apps.yaml` — sonrası tamamen `git push`.
- ⚠️ **Sürüm borcu:** addon **v2.7.2** kurdu (Haziran 2023 — bilinen CVE'leri var).
  Bootstrap için yeterli; ancak UI internete (NPM arkasına) açılmadan önce resmi Helm
  chart'ıyla güncel sürüme taşınmalı. Bilinçli olarak ertelendi.

## 7. Host Docker + NGINX Proxy Manager + UFW

Mimari (proje kararı): cluster'da Ingress yok; TLS ve host bazlı yönlendirme **host'ta
Docker ile çalışan NPM'de**. Servisler NodePort, NPM onlara reverse proxy yapar:

```
Internet → host:443 (NPM) → MicroK8s NodePort → Service → Pod
```

- **Docker Engine** resmi apt deposundan kuruldu (`docker-ce` + compose plugin).
  Bu Docker MicroK8s'ten bağımsızdır (MicroK8s kendi containerd'sini kullanır).
- **NPM** `/opt/npm/docker-compose.yaml` ile ayağa kalktı (`jc21/nginx-proxy-manager`,
  `restart: unless-stopped`). **Admin UI (81) bilinçli olarak `127.0.0.1`'e bağlı** —
  internete açık NPM paneli klasik saldırı yüzeyidir. Panele erişim SSH tüneliyle:
  ```bash
  ssh -L 8181:localhost:81 deephorizon@<sunucu-ip>   # → http://localhost:8181
  ```
  İlk giriş (`admin@example.com / changeme`) yapıldı, kimlik bilgileri değiştirildi ✓
- **UFW:** dışarıya yalnızca 22/80/443; default deny (incoming + routed).
  ```bash
  sudo ufw allow OpenSSH && sudo ufw allow 80/tcp && sudo ufw allow 443/tcp
  sudo ufw enable    # allow OpenSSH'in İŞLENDİĞİNİ görmeden enable etme — kilit riski
  ```
  İlk doğrulamada pod'lar Running + pod içinden `nslookup` cevap verdi; ancak bu test
  yetersizmiş — **pod→host trafiği (API server) sessizce kırılmıştı** (bkz. Hata #7).
  Kalıcı çözüm olarak Calico arayüzleri UFW'den muaf tutuldu:
  ```bash
  sudo ufw allow in on cali+ && sudo ufw allow out on cali+
  sudo ufw allow in on vxlan.calico && sudo ufw allow out on vxlan.calico
  ```
- **UFW'nin yan etkisi (tasarım gereği):** NodePort'lar (örn. Argo CD 30443) dışarıdan
  artık erişilemez. Domain + NPM proxy host'ları gelene kadar UI erişimi SSH tüneliyle:
  `ssh -L 8443:<node-ip>:30443 <kullanici>@<sunucu-ip>` → `https://localhost:8443`.
- **Docker + UFW uyarısı:** Docker publish ettiği portları iptables'a doğrudan yazar,
  UFW'yi bypass edebilir. Host'ta yeni container yayınlarken en basit korunma
  `127.0.0.1:` bind'ıdır (admin portunda yapıldığı gibi).

---

## Karşılaşılan Hatalar: Sebep ve Çözüm

### #1 — `mokutil --sb-state` → `EFI variables are not supported on this system`

- **Neden:** Sunucu UEFI değil **legacy BIOS** modunda boot ediyor. Secure Boot bir
  UEFI özelliğidir; BIOS modunda hiç yoktur.
- **Çözüm:** Çözüm gerekmedi — bu bir hata değil. Secure Boot olmadığı için imzasız
  kernel modülü (NVIDIA sürücüsü) engelsiz yüklenir; `SecureBoot disabled` ile aynı sonuç.

### #2 — `helm repo add` → `404 Not Found` (sealed-secrets index.yaml)

- **Belirti:** `https://bitnami-labs.github.io/sealed-secrets` adresi 404 döndü.
- **Neden:** Sealed Secrets projesi GitHub'da **`bitnami-labs` org'undan `bitnami`
  org'una taşındı** (2026). Eski GitHub Pages Helm repo'su yayından kalktı; internetteki
  eski dokümanların çoğu hâlâ eski URL'i gösteriyor.
- **Çözüm:** Güncel adresler — Helm repo: `https://bitnami.github.io/sealed-secrets`,
  kubeseal release'leri: `github.com/bitnami/sealed-secrets`.

### #3 — `argocd repo add` → `Argo CD server address unspecified`

- **Neden:** CLI hangi sunucuya konuşacağını `argocd login` ile öğrenir; login hiç
  yapılmamıştı.
- **Çözüm:** Önce login (bkz. #4/#5). Alternatif: CLI'sız declarative repo Secret'ı
  (§6'daki YAML) — login gerektirmez.

### #4 — `argocd login localhost:30443` → `gRPC connection not ready: context deadline exceeded`

- **Neden:** kube-proxy NodePort'ları varsayılan olarak **`127.0.0.1`'e bağlamaz**;
  `localhost` üzerinden NodePort'a erişilemez.
- **Çözüm:** Node IP kullan (`kubectl get nodes -o wide` → INTERNAL-IP). Tek başına
  yetmedi — bkz. #5.

### #5 — Node IP + `--grpc-web` ile bile `context deadline exceeded`

- **Neden (asıl kök sebep):** **CLI/server major sürüm farkı.** CLI `releases/latest`'ten
  indirilmişti (v3.x, 2026); MicroK8s community addon'unun kurduğu server ise **v2.7.2**
  (2023). v3 CLI, v2.7 server ile gRPC el sıkışmasını tamamlayamıyor. (Server'ın sağlıklı
  olduğu `curl -skI https://<ip>:30443` → `200 OK` ile teyit edilmişti — sorun hep
  istemci tarafındaydı.)
- **Çözüm:** Server ile **aynı sürüm** CLI kuruldu (v2.7.2) + `--grpc-web` bayrağı
  (NodePort bağlantısı HTTP/1.1'de kaldığından saf gRPC yerine tünellenmiş gRPC).
  Login başarılı, `argocd repo list` → `Successful`.
- **Ders:** `latest` CLI indirme refleksi sürümlü altyapıda tuzak; server sürümünü
  `kubectl -n argocd get deploy argo-cd-argocd-server -o jsonpath='{.spec.template.spec.containers[0].image}'`
  ile öğrenip eşleştir.

### #6 — GitHub'da deploy key eklenemedi

- **Belirti:** Runbook "Settings → Deploy keys" dedi; kullanıcı Settings →
  "Secrets and variables → Agents" sayfasında aradı (yanlış yer — orası Actions/Copilot
  secret'ları içindir), sonra Deploy keys sayfasına yetkisi olmadığı anlaşıldı.
- **Neden:** Deploy keys (`github.com/<org>/<repo>/settings/keys`) yalnızca **repo
  admin'lerine** görünür; kullanıcının bu repoda admin yetkisi yok.
- **Çözüm:** Gerek kalmadı — repo **public** çıktı (kimlik doğrulamasız API 200 dönüyor).
  Argo CD public repo'yu credential'sız okur; bağlantı declarative Secret ile tanımlandı.
  Repo ileride private olursa: read-only deploy key üretilir, public key repo admin'ine
  ekletilir (public key gizli değildir, rahatça iletilebilir).

### #7 — Argo CD `ComparisonError: dial tcp 10.152.183.1:443: i/o timeout` (UFW sonrası)

- **Belirti:** App-of-apps uygulandıktan sonra root Application `Sync Status: Unknown`
  kaldı; koşullarda API server'a (`10.152.183.1:443` — `kubernetes.default` ClusterIP'si)
  i/o timeout. UFW etkinleştirilirken yapılan DNS testi geçtiği halde ortaya çıktı.
- **Neden:** UFW'nin `default deny (incoming)` kuralı **pod→host** trafiğini kesiyor.
  Pod'dan API server'a giden paket host'un INPUT zincirine düşer (API server host'ta
  çalışan bir süreçtir) ve UFW reddeder. DNS testinin geçmesi yanıltıcıydı: o
  **pod→pod** trafiğiydi (FORWARD zinciri — Calico'nun kuralları UFW'den önce kabul
  eder). UFW'den önce kurulmuş bileşenler mevcut bağlantıları üzerinden bir süre
  çalışmaya devam ettiği için sorun gecikmeli göründü.
- **Çözüm:** Calico arayüzlerinden gelen trafik UFW'den muaf tutuldu (`ufw allow in/out
  on cali+` ve `vxlan.calico`). Bu, dışarıya port açmaz — internete bakan yüzey hâlâ
  22/80/443.
- **Ders:** UFW sonrası cluster sağlığını yalnızca pod→pod DNS ile değil, **pod→API
  server** ile de test et: pod içinden `kubernetes.default.svc.cluster.local:443`'e
  bağlantı denemesi veya herhangi bir controller'ın loglarında timeout kontrolü.

---

## Bitiş Durumu ve Sıradaki İşler

**Sunucu platform-hazır:** `nvidia-smi` ✓ · MicroK8s v1.32.13 `Ready` ✓ ·
`nvidia.com/gpu: 1` + `Test PASSED` ✓ · Sealed Secrets (yedekli) ✓ · Argo CD + repo ✓ ·
NPM 80/443 ✓ · UFW ✓

| Sıradaki iş | Not |
|:---|:---|
| `infra/k8s/` manifest'leri | Namespace'ler, app-of-apps, ilk servis (MinIO mantıklı başlangıç) — GitOps akışının uçtan uca testi |
| Argo CD sürüm yükseltme | v2.7.2 → güncel; **UI internete açılmadan önce** (resmi Helm chart'ına geçilerek) |
| Secret'ların seal'lenmesi | Root README "Secret Inventory" envanteri → `infra/k8s/secrets/` |
| Domain + NPM proxy host'ları | Domain alınınca: A kaydı → sunucu IP; her servis için NodePort→domain eşlemesi + Let's Encrypt; sonrasında SSH tünellerine gerek kalmaz |
| GPU contention politikası | Manifest'lerde "default mode": training öncesi inference `replicas: 0` (root README) |
