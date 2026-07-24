package grpc_client

import (
	"context"
	"fmt"
	"sync/atomic"

	pb "github.com/Octapull/deephorizon/services/api/internal/pb/deephorizon/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// poolSize is the number of independent gRPC connections kept open to the
// inference service. gRPC-Go already multiplexes many RPCs over a single
// HTTP/2 connection, so this isn't about throughput — it's so one
// connection stuck reconnecting can't stall every in-flight request;
// round-robin just routes around it onto a healthy sibling.
const poolSize = 4

// retryServiceConfig retries only on UNAVAILABLE, the one status code gRPC
// guarantees means the request never reached the server (a connection-level
// failure before the RPC started). Retrying on other codes such as
// DEADLINE_EXCEEDED risks re-running inference that already started
// server-side, so they're deliberately excluded.
const retryServiceConfig = `{
	"methodConfig": [{
		"name": [{"service": "deephorizon.v1.InferenceService"}],
		"retryPolicy": {
			"MaxAttempts": 4,
			"InitialBackoff": "0.2s",
			"MaxBackoff": "3s",
			"BackoffMultiplier": 2.0,
			"RetryableStatusCodes": ["UNAVAILABLE"]
		}
	}]
}`

type Client struct {
	conns    []*grpc.ClientConn
	services []pb.InferenceServiceClient
	next     atomic.Uint64
}

// New dials a pool of poolSize connections to address. All connections
// target the same address; see poolSize for why a pool exists at all.
func New(address string) (*Client, error) {
	c := &Client{
		conns:    make([]*grpc.ClientConn, 0, poolSize),
		services: make([]pb.InferenceServiceClient, 0, poolSize),
	}

	for i := 0; i < poolSize; i++ {
		conn, err := grpc.NewClient(
			address,
			grpc.WithTransportCredentials(insecure.NewCredentials()),
			grpc.WithDefaultServiceConfig(retryServiceConfig),
		)
		if err != nil {
			c.Close()
			return nil, fmt.Errorf("gRPC bağlantısı kurulamadı: %w", err)
		}
		c.conns = append(c.conns, conn)
		c.services = append(c.services, pb.NewInferenceServiceClient(conn))
	}

	return c, nil
}

// pick round-robins across the connection pool.
func (c *Client) pick() pb.InferenceServiceClient {
	i := c.next.Add(1) % uint64(len(c.services))
	return c.services[i]
}

func (c *Client) Close() {
	for _, conn := range c.conns {
		conn.Close()
	}
}

func (c *Client) Health(ctx context.Context) (*pb.HealthResponse, error) {
	return c.pick().Health(ctx, &pb.HealthRequest{})
}

func (c *Client) ListModels(ctx context.Context) (*pb.ListModelsResponse, error) {
	return c.pick().ListModels(ctx, &pb.ListModelsRequest{})
}

func (c *Client) Enhance(ctx context.Context, req *pb.EnhanceRequest) (*pb.EnhanceResponse, error) {
	return c.pick().Enhance(ctx, req)
}

func (c *Client) EnhanceBatch(ctx context.Context, req *pb.EnhanceBatchRequest) (*pb.EnhanceBatchResponse, error) {
	return c.pick().EnhanceBatch(ctx, req)
}
