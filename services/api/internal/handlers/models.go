package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

func (h *Handler) ListModels(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"models": []gin.H{},
	})
}

func (h *Handler) GetModel(c *gin.Context) {
	modelID := c.Param("id")
	c.JSON(http.StatusOK, gin.H{
		"id":           modelID,
		"architecture": "unknown",
		"status":       "not connected yet",
	})
}