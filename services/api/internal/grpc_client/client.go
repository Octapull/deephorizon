package grpc_client

import (
	"context"
	"fmt"

	pb "github.com/Octapull/deephorizon/services/api/internal/pb/deephorizon/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type Client struct {
	conn    *grpc.ClientConn
	service pb.InferenceServiceClient
}

func New(address string) (*Client, error) {
	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("gRPC bağlantısı kurulamadı: %w", err)
	}

	return &Client{
		conn:    conn,
		service: pb.NewInferenceServiceClient(conn),
	}, nil
}

func (c *Client) Close() {
	c.conn.Close()
}

func (c *Client) Health(ctx context.Context) (*pb.HealthResponse, error) {
	return c.service.Health(ctx, &pb.HealthRequest{})
}

func (c *Client) ListModels(ctx context.Context) (*pb.ListModelsResponse, error) {
	return c.service.ListModels(ctx, &pb.ListModelsRequest{})
}

func (c *Client) Enhance(ctx context.Context, req *pb.EnhanceRequest) (*pb.EnhanceResponse, error) {
	return c.service.Enhance(ctx, req)
}

func (c *Client) EnhanceBatch(ctx context.Context, req *pb.EnhanceBatchRequest) (*pb.EnhanceBatchResponse, error) {
	return c.service.EnhanceBatch(ctx, req)
}