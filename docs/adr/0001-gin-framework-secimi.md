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
