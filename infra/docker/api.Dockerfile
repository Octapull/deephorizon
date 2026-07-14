# Build stage
FROM golang:1.25-alpine AS builder

WORKDIR /app

COPY services/api/go.mod services/api/go.sum ./
RUN go mod download

COPY services/api/ .
RUN go build -o server cmd/server/main.go

# Runtime stage
FROM alpine:latest

WORKDIR /app

COPY --from=builder /app/server .

EXPOSE 8080

CMD ["./server"]