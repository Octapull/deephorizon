package handlers

import (
	"context"
	"log"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

// Health — API + inference server sağlık kontrolü
//
// HTTP GET /health
// Response: {"status", "inference_server", "gpu_available", "detail"}
func (h *Handler) Health(c *gin.Context) {
	// gRPC client yoksa sadece API sağlığını döndür
	if h.GRPCClient == nil {
		c.JSON(http.StatusOK, gin.H{
			"status":            "ok",
			"inference_server":  "disconnected",
			"gpu_available":     false,
			"detail":            "gRPC client not connected",
		})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	resp, err := h.GRPCClient.Health(ctx)
	if err != nil {
		log.Printf("gRPC Health failed: %v", err)
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"status":           "degraded",
			"inference_server": "unreachable",
			"detail":           err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":           "ok",
		"inference_server": "connected",
		"gpu_available":    resp.GpuAvailable,
		"detail":           resp.Detail,
	})
}
