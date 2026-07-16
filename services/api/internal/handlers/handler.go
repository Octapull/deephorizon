package handlers

import (
	grpcclient "github.com/Octapull/deephorizon/services/api/internal/grpc_client"
)

type Handler struct {
	GRPCClient *grpcclient.Client
}

func New(grpcClient *grpcclient.Client) *Handler {
	return &Handler{
		GRPCClient: grpcClient,
	}
}