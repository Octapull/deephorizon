# Monitoring — Prometheus + Grafana

Cluster, node ve **GPU** metriklerini toplar (Prometheus), gösterir (Grafana).
Kurulum GitOps ile: `infra/k8s/monitor/`, `apps/monitor.yaml`, namespace
`deephorizon-monitor`.

## Erişim

| | |
|:---|:---|
| **Grafana (LAN)** | `http://10.10.1.132:30030` |
| **Kullanıcı** | `admin` — parola DevOps'ta (parola kasası) |
| **Prometheus** | ClusterIP (dışarı açılmaz); Grafana'ya datasource olarak önceden bağlı |

UFW 30030'u yalnızca yerel subnet'e açar. İleride NPM arkasına domain'le alınabilir.

## Ne izler

| Kaynak | Ne verir |
|:---|:---|
| **node-exporter** | node CPU / RAM / disk / ağ |
| **kube-state-metrics** | pod/deployment/PVC/job durumları (restart, hazır replika…) |
| **cAdvisor** (kubelet) | konteyner kaynak kullanımı |
| **dcgm-exporter** (GPU operator) | L40S kullanım / VRAM / sıcaklık / güç |
| **anotasyon keşfi** | `prometheus.io/scrape: "true"` taşıyan her pod (api, inference) |

> dcgm-exporter anotasyon taşımadığı için özel bir scrape job'ı ile toplanır
> (`gpu-operator-resources/nvidia-dcgm-exporter`). api/inference deploy olunca
> anotasyonla otomatik girer — ekstra iş yok.

## Dashboard'lar

- **DeepHorizon - Cluster** — koddan otomatik gelir (`infra/k8s/monitor/dashboards/deephorizon-cluster.json`,
  Grafana provisioning). Pod/CPU/RAM/restart + deployment durumu, bizim etiket şemamıza göre.
- **GPU + Node** — topluluk panoları, repoya gömülmedi (JSON büyük). Deploy sonrası
  **elle import** (Grafana → New → Import → ID → datasource `Prometheus`):
  - **12239** — NVIDIA DCGM (GPU: kullanım, VRAM, sıcaklık, güç)
  - **1860** — Node Exporter Full (CPU / RAM / disk / ağ)

## Grafana admin parolası — elle (Git'e girmez)

Grafana pod'u `grafana-admin` Secret'ı olmadan başlamaz. Secret'lar Git dışında;
SealedSecret ile bir kez uygulanır:

```bash
# 1. Duz secret taslagi (commit etme)
kubectl create secret generic grafana-admin \
  --namespace deephorizon-monitor \
  --from-literal=admin-password='<guclu-parola>' \
  --dry-run=client -o yaml > /tmp/grafana-admin.yaml

# 2. Muhurle (namespace scope zorunlu)
kubeseal -n deephorizon-monitor -o yaml \
  < /tmp/grafana-admin.yaml > /tmp/grafana-admin-sealed.yaml

# 3. Cluster'a uygula (Git'e DEGIL)
kubectl apply -f /tmp/grafana-admin-sealed.yaml

# 4. Temizle; muhurlu YAML'i parola kasasina koy
shred -u /tmp/grafana-admin.yaml /tmp/grafana-admin-sealed.yaml
```

> **Sync sırası:** Grafana bu secret'a bağlı. Argo CD namespace'i oluşturduktan
> sonra secret'ı apply et; gelene kadar Grafana `CreateContainerConfigError`'da
> bekler, secret gelince kendi düzelir (takılırsa `kubectl delete pod -n
> deephorizon-monitor -l app=grafana`).

## DevOps notları

| | |
|:---|:---|
| Manifest'ler | `infra/k8s/monitor/` (kustomize) |
| Argo CD app | `infra/k8s/apps/monitor.yaml` |
| Grafana NodePort | `30030` (UFW: `ufw allow from 10.10.1.0/24 to any port 30030 proto tcp`) |
| Secret | `grafana-admin` (SealedSecret, Git dışı) |
| Dashboard'lar | `infra/k8s/monitor/dashboards/*.json` → `grafana-dashboards` ConfigMap (generator) |

**Yeni scrape hedefi eklemek:** pod'a `prometheus.io/scrape: "true"` +
`prometheus.io/port: "<port>"` anotasyonu koy → otomatik toplanır. Anotasyon
konamıyorsa (üçüncü parti) `prometheus.yaml`'e özel bir job ekle (dcgm örneği gibi),
yeniden render/apply et.

**Yeni dashboard eklemek:** JSON'ı `infra/k8s/monitor/dashboards/`'a koy →
`kustomization.yaml`'deki `configMapGenerator.files` listesine ekle → Grafana
provider onu otomatik yükler.
