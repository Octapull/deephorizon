package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

func (h *Handler) Metrics(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status": "metrics not configured yet",
	})
}