# Monitoring — Prometheus + Grafana

Cluster, node ve **GPU (L40S)** metriklerini toplar (Prometheus), gösterir (Grafana).
**Sunucuda elle kuruldu** (`kubectl apply`) — Argo CD/GitOps'a bağlı değil.
Namespace: `deephorizon-monitor`.

## Erişim

| | |
|:---|:---|
| **Grafana (LAN)** | `http://10.10.1.132:30030` |
| **Kullanıcı** | `admin` — parola DevOps'ta (parola kasası) |
| **Prometheus** | ClusterIP (dışarı açılmaz); Grafana'ya datasource olarak önceden bağlı |

## Ne izler

| Kaynak | Ne verir |
|:---|:---|
| **node-exporter** | node CPU / RAM / disk / ağ |
| **kube-state-metrics** | pod/deployment/PVC/job durumları (restart, hazır replika…) |
| **cAdvisor** (kubelet) | konteyner kaynak kullanımı |
| **dcgm-exporter** (GPU operator) | L40S kullanım / VRAM / sıcaklık / güç |
| **anotasyon keşfi** | `prometheus.io/scrape: "true"` taşıyan her pod (api, inference) |

> dcgm-exporter anotasyon taşımadığı için Prometheus config'inde özel bir scrape
> job'ı ile toplanır (`gpu-operator-resources/nvidia-dcgm-exporter`). api/inference
> deploy olunca anotasyonla otomatik girer.

## Dashboard'lar

- **DeepHorizon - Cluster** — ConfigMap ile provision edilir (Grafana açılışta otomatik
  yükler). Pod/CPU/RAM/restart + deployment durumu, bizim etiket şemamıza göre yazıldı.
- **GPU + Node** — topluluk panoları, elle import: Grafana → New → Import → ID gir →
  datasource `Prometheus`:
  - **12239** — NVIDIA DCGM (GPU: kullanım, VRAM, sıcaklık, güç)
  - **1860** — Node Exporter Full (CPU / RAM / disk / ağ)

## Grafana admin parolası (SealedSecret)

Grafana pod'u `grafana-admin` Secret'ı olmadan başlamaz. Parola repoda değil; güçlü
bir parolayla bir kez uygulanır:

```bash
PW=$(openssl rand -base64 24); echo "$PW"        # <- parola kasasına kaydet
kubectl create secret generic grafana-admin -n deephorizon-monitor \
  --from-literal=admin-password="$PW" --dry-run=client -o yaml \
  | kubeseal -n deephorizon-monitor -o yaml | kubectl apply -f -
```
> Secret gelene kadar Grafana `CreateContainerConfigError`'da bekler, gelince kendi
> düzelir (takılırsa `kubectl delete pod -n deephorizon-monitor -l app=grafana`).

## Kurulum ve işletim

- **Deploy:** Manifest'ler **sunucuda elle** uygulanır (`kubectl apply -k <klasör>`).
  **Argo CD yönetmiyor**, GitOps repo'suna girmez — manifest'ler DevOps'ta tutulur.
- **UFW:** Grafana NodePort 30030 yalnızca yerel subnet'e açık:
  `sudo ufw allow from 10.10.1.0/24 to any port 30030 proto tcp`
- **Yeni scrape hedefi:** pod'a `prometheus.io/scrape: "true"` + `prometheus.io/port:
  "<port>"` anotasyonu koy → otomatik toplanır. Anotasyon konamıyorsa (üçüncü parti)
  Prometheus config'ine özel bir job eklenir (dcgm örneği gibi).
- **Güncelleme:** manifest değişince sunucuda tekrar `kubectl apply` + gerekiyorsa
  `kubectl rollout restart deployment/<ad> -n deephorizon-monitor`.

## Durum

Sunucuda kurulu ve doğrulandı: 5 scrape hedefi de `up` (GPU/dcgm dahil), Grafana +
provision edilen cluster panosu canlı.
