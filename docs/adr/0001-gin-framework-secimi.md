# ADR 0001 — HTTP Framework Seçimi: Gin

## Durum
Kabul edildi — Hafta 1

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
- Echo'nun daha esnek middleware sistemi kullanılamayacak
- Tüm handler'lar `*gin.Context` alır
- OpenAPI üretimi için swaggo/swag entegre edilecek (H7)