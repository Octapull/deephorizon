package main

import (
	"log"

	grpcclient "github.com/Octapull/deephorizon/services/api/internal/grpc_client"
	"github.com/Octapull/deephorizon/services/api/internal/handlers"
	"github.com/gin-gonic/gin"
)

func main() {
	grpcClient, err := grpcclient.New("localhost:50051")
	if err != nil {
		log.Printf("gRPC bağlantısı kurulamadı (mock modda çalışılıyor): %v", err)
	}
	if grpcClient != nil {
		defer grpcClient.Close()
	}

	h := handlers.New(grpcClient)

	r := gin.Default()

	r.GET("/health", h.Health)
	r.GET("/models", h.ListModels)
	r.GET("/models/:id", h.GetModel)
	r.POST("/enhance", h.Enhance)
	r.POST("/enhance/batch", h.EnhanceBatch)
	r.GET("/enhance/:job_id", h.GetJob)
	r.GET("/metrics", h.Metrics)

	r.Run(":8080")
}