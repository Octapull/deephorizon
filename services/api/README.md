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

## Generate the OpenAPI spec

`api/openapi.yaml` is generated from `swaggo/swag` annotations on the handlers
(see `// @Summary` etc. comments). `swag` only emits Swagger 2.0, so the
output is converted to OpenAPI 3.0 with `swagger2openapi`:

```bash
cd services/api
swag init -g cmd/server/main.go --output api --outputTypes json
npx --yes swagger2openapi api/swagger.json -o api/openapi.yaml --yaml
rm api/swagger.json
```

## Configuration (env vars)

| Var | Default | Purpose |
|:---|:---|:---|
| `GRPC_ADDR` | `localhost:50051` | Inference service address |
| `REDIS_ADDR` | `localhost:6379` | Job store backend |
| `REDIS_PASSWORD` | *(empty)* | Redis auth |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated list of allowed browser origins (the frontend's dev origin) |
| `MAX_ENHANCE_MB` | `25` | Max request body size for `POST /enhance` (single image) |
| `MAX_BATCH_MB` | `200` | Max request body size for `POST /enhance/batch` (up to 10 images) |

On `SIGINT`/`SIGTERM` the server stops accepting new requests, waits up to
15s for in-flight HTTP requests to finish, then up to 20s for any
still-running background `/enhance` jobs to finish before closing the gRPC
client and Redis connection.
