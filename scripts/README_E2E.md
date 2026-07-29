# End-to-End Integration Test

Bu script, DeepHorizon inference pipeline'ının tamamını test eder:

```
[Checkpoint] → [ONNX Export] → [gRPC Server] → [Client RPC] → [Validation]
```

## Kullanım

### Tam Pipeline (Export + Server + Test)

```bash
# Pix2Pix GAN checkpoint'i ile
python scripts/e2e_integration_test.py \
    --checkpoint checkpoints/best_model.pt \
    --model-type pix2pix_generator

# U-Net baseline checkpoint'i ile
python scripts/e2e_integration_test.py \
    --checkpoint checkpoints/unet_best.pt \
    --model-type unet
```

### Mevcut ONNX Dosyası ile

```bash
python scripts/e2e_integration_test.py \
    --onnx-path exports/deephorizon_generator.onnx
```

### Sadece Export (Server atlama)

```bash
python scripts/e2e_integration_test.py \
    --checkpoint checkpoints/best_model.pt \
    --skip-server
```

### Sonuçları JSON olarak kaydet

```bash
python scripts/e2e_integration_test.py \
    --checkpoint checkpoints/best_model.pt \
    --output-json e2e_results.json
```

## Test Phases

| Phase | Açıklama |
|---|---|
| **1. Export** | Checkpoint → ONNX (validation opsiyonel) |
| **2. Server Startup** | Python gRPC server'ı subprocess olarak başlat |
| **3. Enhance RPC** | Tek görüntü inference + response validation |
| **4. EnhanceBatch RPC** | Batch inference + her response validation |
| **5. ListModels RPC** | Registry'deki modelleri listele |
| **6. Health RPC** | Server sağlık kontrolü |

## Validation Checks

Her RPC response'u için:

| Check | Açıklama |
|---|---|
| `status_ok` | `JOB_STATUS_COMPLETED` (enum=3) |
| `mime_ok` | MIME type `image/` prefix'i ile başlıyor |
| `size_ok` | Output image ≥ 100 bytes |
| `has_metrics` | `inference_time_ms > 0` |

## Exit Codes

| Code | Anlam |
|---|---|
| 0 | Tüm testler geçti |
| 1 | Bir veya daha fazla test başarısız |
| 2 | Setup hatası (missing dependency, file not found, vb.) |

## Gereksinimler

- `torch`, `torchvision` (model export için)
- `onnxruntime` (validation için — opsiyonel)
- `grpcio`, `protobuf` (gRPC client için)
- `Pillow` (PNG encode/decode için)
- `numpy` (sentetik veri için)

Tüm bağımlılıklar `requirements/serving.txt` ve `requirements/ml.txt`'de mevcut.

## Synthetic Test Data

Script, gerçek veriye ihtiyaç duymadan test yapabilmek için **sentetik black-hole benzeri görüntüler** üretir:

- Merkezi parlak nokta (akresyon diski)
- Çevresel halka (foton halkası)
- Gaussian gürültü (teleskop gürültüsü simülasyonu)

Bu sayede CI/CD pipeline'larında veri olmadan da test çalıştırılabilir.

## Örnek Çıktı

```
============================================================
DeepHorizon End-to-End Integration Test
============================================================
  Checkpoint: checkpoints/best_model.pt
  Model type: pix2pix_generator
  Model ID: e2e-test-model

============================================================
PHASE 1: ONNX Export
============================================================
  Checkpoint: checkpoints/best_model.pt
  Model type: pix2pix_generator
  Input shape: (1, 1, 64, 64)
  ✓ ONNX exported: checkpoints/best_model.onnx
  ✓ Metadata: checkpoints/best_model.onnx.json

============================================================
PHASE 2: Server Startup
============================================================
  ✓ Server started on port 51234

============================================================
PHASE 3: Enhance RPC Test
============================================================
  Input: (64, 64) float32 → 234 bytes PNG
  Round-trip: 45.2ms
  Status: True (3)
  MIME: True (image/png)
  Size: True (312 bytes)
  Metrics: True (inference_time_ms=42)
  Output shape: (64, 64)
  ✓ Enhance RPC test PASSED

...

============================================================
TEST SUMMARY
============================================================
  ✓ PASS  export
  ✓ PASS  server_start
  ✓ PASS  enhance
  ✓ PASS  enhance_batch
  ✓ PASS  list_models
  ✓ PASS  health

Overall: ✓ ALL TESTS PASSED
```

## CI/CD Integration

GitHub Actions örneği:

```yaml
- name: Run E2E integration test
  run: |
    python scripts/e2e_integration_test.py \
      --checkpoint tests/fixtures/dummy_model.pt \
      --model-type pix2pix_generator \
      --output-json e2e_results.json

- name: Upload test results
  uses: actions/upload-artifact@v4
  with:
    name: e2e-results
    path: e2e_results.json
```

## Troubleshooting

### "Proto stubs not found"

```bash
cd proto && buf generate && cd ..
```

### "ONNX model not found"

Checkpoint path'ini kontrol edin veya `--skip-server` ile sadece export testi yapın.

### "gRPC server failed to start"

- Port çakışması olabilir — script otomatik free port bulur
- `torch` import hatası — `pip install -r requirements/ml.txt`
- Log dosyasını kontrol edin: `--log-file server.log` (henüz eklenmedi)

### "Enhance RPC timeout"

- Model çok büyük olabilir — `--input-shape 1 1 32 32` ile küçültün
- GPU yoksa CPU inference yavaş — `--device cpu` (server tarafında)
