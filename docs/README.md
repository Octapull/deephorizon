# docs

Project documentation that does not belong in source-level READMEs.

## Layout

| Path | Contents |
|:---|:---|
| [`SOZLUK.md`](SOZLUK.md) | **Teknik sözlük (TR)** — proto, gitkeep, gRPC, Argo CD, MicroK8s, MLflow, PSF, UV-plane vb. her terim. **Stajyer ilk gün açacak.** |
| [`DATA.md`](DATA.md) | **Veri deposu rehberi (TR)** — MinIO bucket düzeni, versiyonlama kuralı, ML tarafı için erişim örnekleri (mc / s3fs), veri yükleme akışı. |
| [`DVC.md`](DVC.md) | **DVC rehberi (TR)** — veri versiyonlama: `dvc-cache` remote'unun kurulumu (`dvc init` + `remote add`), push/pull kullanımı, eski sürüme dönme. **DVC kullanacak kişinin adresi.** |
| [`DEVOPS.md`](DEVOPS.md) | **DevOps kurulum günlüğü (TR)** — GPU sunucu bootstrap'ının tamamı: NVIDIA sürücü, MicroK8s + GPU addon, Sealed Secrets, Argo CD, NPM, UFW; karşılaşılan hatalar, sebepleri ve çözümleri. |
| `adr/` | Architecture Decision Records — one file per non-trivial decision |
| `runbooks/` | On-call / incident playbooks (one per failure mode) |

## ADR Format

```
adr/NNNN-short-title.md
```

Each ADR has: **Context** · **Decision** · **Consequences** · **Status** (proposed / accepted / superseded).
