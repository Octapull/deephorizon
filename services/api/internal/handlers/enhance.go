package handlers

import (
	"bytes"
	"context"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	pb "github.com/Octapull/deephorizon/services/api/internal/pb/deephorizon/v1"
)

// EnhanceRequest — multipart/form-data form binding
type EnhanceRequest struct {
	ModelID      string `form:"model_id" binding:"required"`
	ScaleFactor  uint32 `form:"scale_factor" binding:"required,oneof=1 2 4"`
	OutputFormat string `form:"output_format" binding:"required,oneof=png jpeg fits"`
}

// gRPC timeout — inference server'ın yanıt vermesi için max süre
const grpcTimeout = 30 * time.Second

// Enhance — tek görüntü inference
//
// HTTP POST /enhance
//   multipart/form-data:
//     - image:        dosya (PNG/JPEG/FITS)
//     - model_id:     string
//     - scale_factor: 1 | 2 | 4
//     - output_format: png | jpeg | fits
//
// Response: EnhanceResponse JSON (image base64 + metrics)
func (h *Handler) Enhance(c *gin.Context) {
	file, err := c.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "image dosyası eksik"})
		return
	}

	var req EnhanceRequest
	if err := c.ShouldBind(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	src, err := file.Open()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "dosya okunamadı"})
		return
	}
	defer src.Close()

	imageBytes, mimeType, width, height, err := readImage(src, file.Filename)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// gRPC client yoksa mock response döndür (graceful degradation)
	if h.GRPCClient == nil {
		c.JSON(http.StatusAccepted, gin.H{
			"job_id":        uuid.New().String(),
			"status":        "JOB_STATUS_QUEUED",
			"model_id":      req.ModelID,
			"scale_factor":  req.ScaleFactor,
			"output_format": req.OutputFormat,
			"note":          "gRPC client not connected — mock response",
		})
		return
	}

	// gRPC Enhance çağrısı
	ctx, cancel := context.WithTimeout(c.Request.Context(), grpcTimeout)
	defer cancel()

	grpcReq := &pb.EnhanceRequest{
		Image: &pb.ImagePayload{
			Data:     imageBytes,
			MimeType: mimeType,
			Width:    uint32(width),
			Height:   uint32(height),
		},
		ModelId:      req.ModelID,
		ScaleFactor:  req.ScaleFactor,
		OutputFormat: req.OutputFormat,
	}

	resp, err := h.GRPCClient.Enhance(ctx, grpcReq)
	if err != nil {
		log.Printf("gRPC Enhance failed: %v", err)
		c.JSON(http.StatusBadGateway, gin.H{
			"error":  "inference server error",
			"detail": err.Error(),
		})
		return
	}

	// Response → JSON
	c.JSON(http.StatusOK, gin.H{
		"job_id":   uuid.New().String(),
		"status":   jobStatusToString(resp.Status),
		"model_id": resp.ModelId,
		"image": gin.H{
			"data":      resp.Image.Data,
			"mime_type": resp.Image.MimeType,
			"width":     resp.Image.Width,
			"height":    resp.Image.Height,
		},
		"metrics": gin.H{
			"psnr":              resp.Metrics.Psnr,
			"ssim":              resp.Metrics.Ssim,
			"lpips":             resp.Metrics.Lpips,
			"inference_time_ms": resp.Metrics.InferenceTimeMs,
		},
	})
}

// EnhanceBatch — batch inference
//
// HTTP POST /enhance/batch
//   multipart/form-data:
//     - images:       birden fazla dosya
//     - model_id:     string
//     - scale_factor: 1 | 2 | 4
//     - output_format: png | jpeg | fits
func (h *Handler) EnhanceBatch(c *gin.Context) {
	form, err := c.MultipartForm()
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "multipart form okunamadı"})
		return
	}

	files := form.File["images"]
	if len(files) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "en az 1 görüntü gerekli"})
		return
	}
	if len(files) > 10 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "en fazla 10 görüntü gönderilebilir"})
		return
	}

	modelID := c.PostForm("model_id")
	if modelID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model_id gerekli"})
		return
	}
	scaleFactor := uint32(1)
	outputFormat := c.DefaultPostForm("output_format", "png")

	// gRPC client yoksa mock response
	if h.GRPCClient == nil {
		jobs := make([]gin.H, 0, len(files))
		for range files {
			jobs = append(jobs, gin.H{
				"job_id": uuid.New().String(),
				"status": "JOB_STATUS_QUEUED",
			})
		}
		c.JSON(http.StatusAccepted, gin.H{
			"jobs": jobs,
			"note": "gRPC client not connected — mock response",
		})
		return
	}

	// gRPC EnhanceBatch çağrısı
	ctx, cancel := context.WithTimeout(c.Request.Context(), grpcTimeout*2)
	defer cancel()

	requests := make([]*pb.EnhanceRequest, 0, len(files))
	for _, file := range files {
		src, err := file.Open()
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "dosya okunamadı: " + file.Filename})
			return
		}
		imageBytes, mimeType, width, height, err := readImage(src, file.Filename)
		src.Close()
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		requests = append(requests, &pb.EnhanceRequest{
			Image: &pb.ImagePayload{
				Data:     imageBytes,
				MimeType: mimeType,
				Width:    uint32(width),
				Height:   uint32(height),
			},
			ModelId:      modelID,
			ScaleFactor:  scaleFactor,
			OutputFormat: outputFormat,
		})
	}

	batchReq := &pb.EnhanceBatchRequest{
		Requests:     requests,
		MaxBatchSize: uint32(len(requests)),
	}

	batchResp, err := h.GRPCClient.EnhanceBatch(ctx, batchReq)
	if err != nil {
		log.Printf("gRPC EnhanceBatch failed: %v", err)
		c.JSON(http.StatusBadGateway, gin.H{
			"error":  "inference server error",
			"detail": err.Error(),
		})
		return
	}

	// Response → JSON
	responses := make([]gin.H, 0, len(batchResp.Responses))
	for _, resp := range batchResp.Responses {
		responses = append(responses, gin.H{
			"status":   jobStatusToString(resp.Status),
			"model_id": resp.ModelId,
			"image": gin.H{
				"data":      resp.Image.Data,
				"mime_type": resp.Image.MimeType,
				"width":     resp.Image.Width,
				"height":    resp.Image.Height,
			},
			"metrics": gin.H{
				"psnr":              resp.Metrics.Psnr,
				"ssim":              resp.Metrics.Ssim,
				"lpips":             resp.Metrics.Lpips,
				"inference_time_ms": resp.Metrics.InferenceTimeMs,
			},
		})
	}

	c.JSON(http.StatusOK, gin.H{
		"jobs": responses,
	})
}

// GetJob — job durumu sorgulama (şimdilik mock — async job queue yok)
func (h *Handler) GetJob(c *gin.Context) {
	jobID := c.Param("job_id")
	c.JSON(http.StatusOK, gin.H{
		"job_id": jobID,
		"status": "JOB_STATUS_COMPLETED",
		"note":   "async job queue not implemented yet — sync mode only",
	})
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// readImage — dosyadan bytes + mime_type + width + height çıkar
func readImage(src io.Reader, filename string) ([]byte, string, int, int, error) {
	data, err := io.ReadAll(src)
	if err != nil {
		return nil, "", 0, 0, err
	}

	mimeType := detectMimeType(filename, data)

	// PNG/JPEG için boyut tespiti
	if mimeType == "image/png" || mimeType == "image/jpeg" {
		img, _, err := image.Decode(bytes.NewReader(data))
		if err != nil {
			return nil, "", 0, 0, err
		}
		bounds := img.Bounds()
		return data, mimeType, bounds.Dx(), bounds.Dy(), nil
	}

	// FITS veya raw float32 — boyut header'dan veya form'dan gelecek
	// Şimdilik 0,0 döndür (Python server FITS header'ı parse eder)
	return data, mimeType, 0, 0, nil
}

// detectMimeType — dosya uzantısı + magic bytes ile MIME tespiti
func detectMimeType(filename string, data []byte) string {
	// Uzantıya göre
	if len(filename) >= 4 {
		ext := filename[len(filename)-4:]
		switch ext {
		case ".png":
			return "image/png"
		case ".jpg", "jpeg":
			return "image/jpeg"
		}
		if len(filename) >= 5 && filename[len(filename)-5:] == ".fits" {
			return "image/fits"
		}
	}

	// Magic bytes
	if len(data) >= 8 {
		// PNG: 89 50 4E 47 0D 0A 1A 0A
		if data[0] == 0x89 && data[1] == 'P' && data[2] == 'N' && data[3] == 'G' {
			return "image/png"
		}
		// JPEG: FF D8 FF
		if data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF {
			return "image/jpeg"
		}
		// FITS: "SIMPLE  ="
		if bytes.HasPrefix(data, []byte("SIMPLE  =")) {
			return "image/fits"
		}
	}

	// Raw float32 — boyut header'dan gelecek
	return "application/octet-stream"
}

// jobStatusToString — proto enum → string
func jobStatusToString(status pb.JobStatus) string {
	switch status {
	case pb.JobStatus_JOB_STATUS_QUEUED:
		return "JOB_STATUS_QUEUED"
	case pb.JobStatus_JOB_STATUS_RUNNING:
		return "JOB_STATUS_RUNNING"
	case pb.JobStatus_JOB_STATUS_COMPLETED:
		return "JOB_STATUS_COMPLETED"
	case pb.JobStatus_JOB_STATUS_FAILED:
		return "JOB_STATUS_FAILED"
	default:
		return "JOB_STATUS_UNSPECIFIED"
	}
}
