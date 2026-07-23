package handlers

import (
	"context"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	pb "github.com/Octapull/deephorizon/services/api/internal/pb/deephorizon/v1"
)

const modelsCallTimeout = 5 * time.Second

func modelInfoJSON(m *pb.ModelInfo) gin.H {
	entry := gin.H{
		"id":           m.GetId(),
		"architecture": m.GetArchitecture(),
		"version":      m.GetVersion(),
	}
	if vm := m.GetValidationMetrics(); vm != nil {
		entry["validation_metrics"] = gin.H{
			"psnr":              vm.GetPsnr(),
			"ssim":              vm.GetSsim(),
			"lpips":             vm.GetLpips(),
			"inference_time_ms": vm.GetInferenceTimeMs(),
		}
	}
	return entry
}

// ListModels godoc
// @Summary      List available models
// @Description  Proxies the ListModels RPC to the inference service.
// @Tags         models
// @Produce      json
// @Success      200  {object}  map[string]interface{}
// @Failure      503  {object}  map[string]interface{}
// @Router       /models [get]
func (h *Handler) ListModels(c *gin.Context) {
	if h.GRPCClient == nil {
		c.JSON(http.StatusOK, gin.H{
			"models": []gin.H{},
			"detail": "inference service bağlantısı yok (mock modda çalışılıyor)",
		})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), modelsCallTimeout)
	defer cancel()

	resp, err := h.GRPCClient.ListModels(ctx)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "inference service ulaşılamadı: " + err.Error()})
		return
	}

	models := make([]gin.H, 0, len(resp.GetModels()))
	for _, m := range resp.GetModels() {
		models = append(models, modelInfoJSON(m))
	}
	c.JSON(http.StatusOK, gin.H{"models": models})
}

// GetModel godoc
// @Summary      Get a single model by id
// @Description  Fetches ListModels from the inference service and filters by id.
// @Tags         models
// @Produce      json
// @Param        id   path      string  true  "Model id"
// @Success      200  {object}  map[string]interface{}
// @Failure      404  {object}  map[string]interface{}
// @Failure      503  {object}  map[string]interface{}
// @Router       /models/{id} [get]
func (h *Handler) GetModel(c *gin.Context) {
	modelID := c.Param("id")

	if h.GRPCClient == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "inference service bağlantısı yok (mock modda çalışılıyor)"})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), modelsCallTimeout)
	defer cancel()

	resp, err := h.GRPCClient.ListModels(ctx)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "inference service ulaşılamadı: " + err.Error()})
		return
	}

	for _, m := range resp.GetModels() {
		if m.GetId() == modelID {
			c.JSON(http.StatusOK, modelInfoJSON(m))
			return
		}
	}
	c.JSON(http.StatusNotFound, gin.H{"error": "model bulunamadı", "id": modelID})
}
