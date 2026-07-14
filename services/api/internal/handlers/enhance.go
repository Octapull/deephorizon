package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

type EnhanceRequest struct {
	ModelID      string `form:"model_id" binding:"required"`
	ScaleFactor  uint32 `form:"scale_factor" binding:"required,oneof=1 2 4"`
	OutputFormat string `form:"output_format" binding:"required,oneof=png fits"`
}

func (h *Handler) Enhance(c *gin.Context) {
	file, err := c.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "image dosyası eksik",
		})
		return
	}

	var req EnhanceRequest
	if err := c.ShouldBind(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": err.Error(),
		})
		return
	}

	src, err := file.Open()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "dosya okunamadı",
		})
		return
	}
	defer src.Close()

	// TODO: gRPC çağrısı buraya gelecek (H6-H7)

	c.JSON(http.StatusAccepted, gin.H{
		"job_id":        uuid.New().String(),
		"status":        "JOB_STATUS_QUEUED",
		"model_id":      req.ModelID,
		"scale_factor":  req.ScaleFactor,
		"output_format": req.OutputFormat,
	})
}

func (h *Handler) EnhanceBatch(c *gin.Context) {
	form, err := c.MultipartForm()
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "multipart form okunamadı",
		})
		return
	}

	files := form.File["images"]
	if len(files) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "en az 1 görüntü gerekli",
		})
		return
	}
	if len(files) > 10 {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "en fazla 10 görüntü gönderilebilir",
		})
		return
	}

	jobs := make([]gin.H, 0, len(files))
	for range files {
		jobs = append(jobs, gin.H{
			"job_id": uuid.New().String(),
			"status": "JOB_STATUS_QUEUED",
		})
	}

	c.JSON(http.StatusAccepted, gin.H{
		"jobs": jobs,
	})
}

func (h *Handler) GetJob(c *gin.Context) {
	jobID := c.Param("job_id")
	c.JSON(http.StatusOK, gin.H{
		"job_id": jobID,
		"status": "JOB_STATUS_QUEUED",
	})
}