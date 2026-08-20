# ADR 0001 — HTTP Framework Seçimi: Gin

## Bağlam
Go API gateway için bir HTTP framework seçilmesi gerekiyordu.
İki aday değerlendirildi: Gin ve Echo.

## Karar
Gin seçildi.

## Gerekçe
- swaggo/swag (OpenAPI üretimi) ve go-playground/validator Gin ile daha yaygın kullanılıyor
- README'de öngörülen bu iki kütüphane Gin ekosisteminde daha olgun
- Topluluk ve kaynak sayısı daha fazla, staj süresi için öğrenme maliyeti düşük

## Sonuçlar
- Tüm handler'lar `*gin.Context` alır
- OpenAPI üretimi için swaggo/swag entegre edilecek 

---

# ADR 0002 — Protobuf Kontratı: EnhanceBatch RPC Eklenmesi


## Bağlam
`POST /enhance/batch` endpoint'i 10 görüntüye kadar toplu istek kabul ediyor.
Proto'da sadece tekli `Enhance` RPC vardı.

## Karar
`EnhanceBatch` RPC proto'ya eklendi.

## Gerekçe
- 10 ayrı gRPC çağrısı yerine tek çağrı ile GPU batch inference yapılabilir
- PyTorch modelleri dynamic batch export destekliyor
- Performans ve GPU verimliliği açısından gerekli

## Sonuçlar
- Python tarafında `EnhanceBatch` implementasyonu yazılacak
- Go tarafında `internal/grpc_client/client.go`'ya `EnhanceBatch` çağrısı eklendi
- `buf lint` temiz, stub'lar yeniden üretildi

---

# ADR 0003 — Handler Yapısı: Dependency Injection


## Bağlam
Handler'lar başlangıçta bağımsız fonksiyonlardı. gRPC istemcisini handler'lara aktarmak için bir yapıya ihtiyaç vardı.

## Karar
Handler'lar `Handler` struct'ına taşındı, gRPC istemcisi constructor ile inject edildi.

## Gerekçe
- Handler'lar gRPC istemcisine bağımlı, bağımlılığı açık hale getirmek test edilebilirliği artırır
- gRPC istemcisi gerçek sunucuya bağlanınca sadece `main.go` değişecek
- Go'da yaygın ve kabul görmüş pattern

## Sonuçlar
- Tüm handler'lar `(h *Handler)` metoduna dönüştürüldü
- `main.go` gRPC istemcisini oluşturup handler'a veriyor
- Mock mod destekleniyor — gRPC bağlantısı kurulamazsa sunucu çalışmaya devam ediyor

---

# ADR 0004 — /enhance için Async Job Akışı ve Redis Job Store

## Bağlam
`POST /enhance` başlangıçta gerçek bir işlem yapmadan her zaman `202 Accepted` +
`JOB_STATUS_QUEUED` dönüyordu; `GET /enhance/:job_id` de her zaman aynı sahte
durumu veriyordu. `common.proto`'daki eski yorum ("Sync /enhance requests jump
straight to COMPLETED") ile de çelişiyordu. Gerçek gRPC çağrısı bağlanınca bu
akışın netleşmesi gerekti.

## Karar
`/enhance` ve `/enhance/batch` tamamen asenkron çalışacak şekilde uygulandı:
istek anında `QUEUED` job_id ile döner, arka planda bir goroutine gRPC
çağrısını yapar, job durumunu `RUNNING` → `COMPLETED`/`FAILED` olarak Redis'e
yazar. Sonuç `GET /enhance/:job_id` ile sorgulanır. Job store için
`infra/k8s/redis/` altında zaten tanımlı olan Redis kullanıldı (in-memory map
yerine).

## Gerekçe
- README'de Intern 6 rolü hem Go gateway'i hem de shared protobuf kontratını
  kapsıyor ve "async job handling" açıkça sorumluluk alanında; roadmap'in
  7. hafta teslimatı da birebir "Real gRPC call from Go → Python, async job
  flow" olarak tanımlı — bu ADR o teslimatı karşılıyor.
- Senkron model proto yorumuyla tutarlı olurdu ama gerçek bir API gateway
  davranışı (büyük/batch görüntülerde timeout riski olmadan hemen 202 dönmek)
  asenkron modelle daha iyi örtüşüyor.
- Redis zaten `infra/k8s/redis/` içinde deploy edilmiş durumda; in-memory map
  yerine bunu kullanmak tek pod ötesinde ölçeklenebilirlik sağlıyor ve
  gelecekte gerçek bir job queue'ya (Redis Streams vb.) geçişi kolaylaştırıyor.

## Sonuçlar
- `common.proto`'daki `JobStatus` yorumu güncellendi, `buf generate` ile Go
  (ve ilk kez Python) stub'ları yeniden üretildi.
- `internal/jobstore/` eklendi: Redis'e TTL'li (24s) JSON olarak yazan basit
  bir job store.
- `internal/metrics/` eklendi: `/metrics` artık gerçek Prometheus text
  formatında (request sayısı/süresi + job sonuç sayaçları) veri veriyor.
- `main.go` `REDIS_ADDR` / `REDIS_PASSWORD` / `GRPC_ADDR` ortam
  değişkenlerini okuyor.
- Riski yok: sadece proto yorumu değişti, wire format aynı kaldı; `buf
  breaking` kontrolüne takılmaz.
