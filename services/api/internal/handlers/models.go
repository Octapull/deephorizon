package handlers

import (
	"context"
	"log"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

// ListModels — inference server'daki tüm modelleri listele
//
// HTTP GET /models
// Response: {"models": [{"id", "architecture", "version", "validation_metrics"}]}
func (h *Handler) ListModels(c *gin.Context) {
	// gRPC client yoksa boş liste döndür
	if h.GRPCClient == nil {
		c.JSON(http.StatusOK, gin.H{
			"models": []gin.H{},
			"note":   "gRPC client not connected",
		})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	resp, err := h.GRPCClient.ListModels(ctx)
	if err != nil {
		log.Printf("gRPC ListModels failed: %v", err)
		c.JSON(http.StatusBadGateway, gin.H{
			"error":  "inference server error",
			"detail": err.Error(),
		})
		return
	}

	// Proto → JSON
	models := make([]gin.H, 0, len(resp.Models))
	for _, m := range resp.Models {
		models = append(models, gin.H{
			"id":           m.Id,
			"architecture": m.Architecture,
			"version":      m.Version,
			"validation_metrics": gin.H{
				"psnr": m.ValidationMetrics.Psnr,
				"ssim": m.ValidationMetrics.Ssim,
			},
		})
	}

	c.JSON(http.StatusOK, gin.H{"models": models})
}

// GetModel — tek model detayı (şimdilik ListModels'dan filtreleme)
//
// HTTP GET /models/:id
func (h *Handler) GetModel(c *gin.Context) {
	modelID := c.Param("id")

	// gRPC client yoksa mock response
	if h.GRPCClient == nil {
		c.JSON(http.StatusOK, gin.H{
			"id":           modelID,
			"architecture": "unknown",
			"status":       "gRPC client not connected",
		})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	resp, err := h.GRPCClient.ListModels(ctx)
	if err != nil {
		log.Printf("gRPC ListModels failed: %v", err)
		c.JSON(http.StatusBadGateway, gin.H{
			"error":  "inference server error",
			"detail": err.Error(),
		})
		return
	}

	// ID'ye göre filtrele
	for _, m := range resp.Models {
		if m.Id == modelID {
			c.JSON(http.StatusOK, gin.H{
				"id":           m.Id,
				"architecture": m.Architecture,
				"version":      m.Version,
				"validation_metrics": gin.H{
					"psnr": m.ValidationMetrics.Psnr,
					"ssim": m.ValidationMetrics.Ssim,
				},
			})
			return
		}
	}

	c.JSON(http.StatusNotFound, gin.H{
		"error":    "model not found",
		"model_id": modelID,
	})
}
