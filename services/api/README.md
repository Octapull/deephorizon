# services/api

Go API gateway. Translates REST requests into gRPC calls to `services/ml/inference_server/`.

**Owner:** Stajyer 6.

## Layout

| Path | Purpose |
|:---|:---|
| `cmd/server/` | `main.go` — process entry point |
| `internal/handlers/` | HTTP handlers: `/enhance`, `/models`, `/health`, `/metrics` |
| `internal/grpc_client/` | gRPC client to the inference service (with retries + pool) |
| `internal/pb/` | Generated protobuf stubs (from `proto/`, do not edit by hand) |
| `api/openapi.yaml` | Generated OpenAPI 3.0 spec |

## Bootstrap

```bash
cd services/api
go mod init github.com/Octapull/deephorizon/services/api
go mod tidy
```

## Generate proto stubs

Run from the repo root:

```bash
cd proto && buf generate
```
