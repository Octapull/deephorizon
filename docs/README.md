# docs

Project documentation that does not belong in source-level READMEs.

## Layout

| Path | Contents |
|:---|:---|
| [`SOZLUK.md`](SOZLUK.md) | **Teknik sözlük (TR)** — proto, gitkeep, gRPC, Argo CD, MicroK8s, MLflow, PSF, UV-plane vb. her terim. **Stajyer ilk gün açacak.** |
| `adr/` | Architecture Decision Records — one file per non-trivial decision |
| `runbooks/` | On-call / incident playbooks (one per failure mode) |

## ADR Format

```
adr/NNNN-short-title.md
```

Each ADR has: **Context** · **Decision** · **Consequences** · **Status** (proposed / accepted / superseded).
