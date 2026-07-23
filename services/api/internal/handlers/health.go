package handlers

import (
	"context"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

const healthCheckTimeout = 3 * time.Second

// Health godoc
// @Summary      Gateway and inference service health
// @Description  Reports the gateway's own status plus, if reachable, the inference service's status via the Health RPC.
// @Tags         health
// @Produce      json
// @Success      200  {object}  map[string]interface{}
// @Router       /health [get]
func (h *Handler) Health(c *gin.Context) {
	if h.GRPCClient == nil {
		c.JSON(http.StatusOK, gin.H{
			"status":            "ok",
			"gpu_available":     false,
			"inference_service": "not connected (mock mode)",
		})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), healthCheckTimeout)
	defer cancel()

	resp, err := h.GRPCClient.Health(ctx)
	if err != nil {
		c.JSON(http.StatusOK, gin.H{
			"status":            "ok",
			"gpu_available":     false,
			"inference_service": "unreachable: " + err.Error(),
		})
		return
	}

	inferenceStatus := "ok"
	if !resp.GetOk() {
		inferenceStatus = "degraded"
	}

	c.JSON(http.StatusOK, gin.H{
		"status":            "ok",
		"gpu_available":     resp.GetGpuAvailable(),
		"inference_service": inferenceStatus,
		"inference_detail":  resp.GetDetail(),
	})
}
