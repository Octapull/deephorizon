package handlers

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"mime/multipart"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"

	"github.com/Octapull/deephorizon/services/api/internal/jobstore"
	"github.com/Octapull/deephorizon/services/api/internal/metrics"
	pb "github.com/Octapull/deephorizon/services/api/internal/pb/deephorizon/v1"
)

const maxBatchImages = 10

// enhanceTimeout bounds a background gRPC call. It is intentionally
// decoupled from the HTTP request context, which ends as soon as we
// respond 202 Accepted.
const enhanceTimeout = 5 * time.Minute

type EnhanceRequest struct {
	ModelID      string `form:"model_id" binding:"required"`
	ScaleFactor  uint32 `form:"scale_factor" binding:"required,oneof=1 2 4"`
	OutputFormat string `form:"output_format" binding:"required,oneof=png fits"`
}

// Enhance godoc
// @Summary      Queue a single-image enhancement job
// @Description  Accepts one image and queues an async super-resolution job. Poll GET /enhance/{job_id} for the result.
// @Tags         enhance
// @Accept       multipart/form-data
// @Produce      json
// @Param        image          formData  file    true  "Image to enhance"
// @Param        model_id       formData  string  true  "Model identifier"
// @Param        scale_factor   formData  int     true  "Scale factor"  Enums(1, 2, 4)
// @Param        output_format  formData  string  true  "Output format"  Enums(png, fits)
// @Success      202  {object}  map[string]interface{}
// @Failure      400  {object}  map[string]interface{}
// @Failure      500  {object}  map[string]interface{}
// @Router       /enhance [post]
func (h *Handler) Enhance(c *gin.Context) {
	var req EnhanceRequest
	if err := c.ShouldBind(&req); err != nil {
		respondBindError(c, err)
		return
	}

	file, err := c.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "image dosyası eksik"})
		return
	}

	data, mimeType, width, height, err := readImageFile(file)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	job := &jobstore.Job{
		ID:           uuid.New().String(),
		Status:       jobstore.StatusQueued,
		ModelID:      req.ModelID,
		ScaleFactor:  req.ScaleFactor,
		OutputFormat: req.OutputFormat,
	}
	if err := h.Jobs.Create(c.Request.Context(), job); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "job kaydedilemedi: " + err.Error()})
		return
	}

	pbReq := &pb.EnhanceRequest{
		Image: &pb.ImagePayload{
			Data:     data,
			MimeType: mimeType,
			Width:    width,
			Height:   height,
		},
		ModelId:      req.ModelID,
		ScaleFactor:  req.ScaleFactor,
		OutputFormat: req.OutputFormat,
	}
	h.wg.Add(1)
	go func() {
		defer h.wg.Done()
		h.runEnhanceJob(job.ID, pbReq)
	}()

	c.JSON(http.StatusAccepted, gin.H{
		"job_id":        job.ID,
		"status":        string(job.Status),
		"model_id":      job.ModelID,
		"scale_factor":  job.ScaleFactor,
		"output_format": job.OutputFormat,
	})
}

// runEnhanceJob calls the inference service in the background and writes
// the outcome back to the job store. It runs on its own context/timeout,
// detached from the request that queued it.
func (h *Handler) runEnhanceJob(jobID string, req *pb.EnhanceRequest) {
	ctx, cancel := context.WithTimeout(context.Background(), enhanceTimeout)
	defer cancel()

	job, err := h.Jobs.Get(ctx, jobID)
	if err != nil {
		return
	}
	job.Status = jobstore.StatusRunning
	_ = h.Jobs.Update(ctx, job)

	if h.GRPCClient == nil {
		job.Status = jobstore.StatusFailed
		job.Error = "inference service bağlantısı yok (mock modda çalışılıyor)"
		_ = h.Jobs.Update(ctx, job)
		metrics.EnhanceJobsTotal.WithLabelValues(string(job.Status)).Inc()
		return
	}

	resp, err := h.GRPCClient.Enhance(ctx, req)
	if err != nil {
		job.Status = jobstore.StatusFailed
		job.Error = err.Error()
		_ = h.Jobs.Update(ctx, job)
		metrics.EnhanceJobsTotal.WithLabelValues(string(job.Status)).Inc()
		return
	}

	applyEnhanceResponse(job, resp)
	_ = h.Jobs.Update(ctx, job)
	metrics.EnhanceJobsTotal.WithLabelValues(string(job.Status)).Inc()
}

func applyEnhanceResponse(job *jobstore.Job, resp *pb.EnhanceResponse) {
	if img := resp.GetImage(); img != nil {
		job.ResultImage = img.GetData()
		job.ResultMime = img.GetMimeType()
	}
	if m := resp.GetMetrics(); m != nil {
		job.Metrics = &jobstore.Metrics{
			PSNR:            m.GetPsnr(),
			SSIM:            m.GetSsim(),
			LPIPS:           m.GetLpips(),
			InferenceTimeMs: m.GetInferenceTimeMs(),
		}
	}
	if resp.GetStatus() == pb.JobStatus_JOB_STATUS_FAILED {
		job.Status = jobstore.StatusFailed
		if job.Error == "" {
			job.Error = "inference service reported failure"
		}
		return
	}
	job.Status = jobstore.StatusCompleted
}

// EnhanceBatch godoc
// @Summary      Queue a batch enhancement job
// @Description  Accepts up to 10 images sharing the same model/scale/format, queues one job per image, and dispatches them as a single EnhanceBatch gRPC call.
// @Tags         enhance
// @Accept       multipart/form-data
// @Produce      json
// @Param        images         formData  file    true  "Images to enhance (max 10)"
// @Param        model_id       formData  string  true  "Model identifier"
// @Param        scale_factor   formData  int     true  "Scale factor"  Enums(1, 2, 4)
// @Param        output_format  formData  string  true  "Output format"  Enums(png, fits)
// @Success      202  {object}  map[string]interface{}
// @Failure      400  {object}  map[string]interface{}
// @Failure      500  {object}  map[string]interface{}
// @Router       /enhance/batch [post]
func (h *Handler) EnhanceBatch(c *gin.Context) {
	var req EnhanceRequest
	if err := c.ShouldBind(&req); err != nil {
		respondBindError(c, err)
		return
	}

	form, err := c.MultipartForm()
	if err != nil {
		respondBindError(c, err)
		return
	}

	files := form.File["images"]
	if len(files) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "en az 1 görüntü gerekli"})
		return
	}
	if len(files) > maxBatchImages {
		c.JSON(http.StatusBadRequest, gin.H{"error": "en fazla 10 görüntü gönderilebilir"})
		return
	}

	jobs := make([]*jobstore.Job, 0, len(files))
	pbRequests := make([]*pb.EnhanceRequest, 0, len(files))

	for _, fh := range files {
		data, mimeType, width, height, err := readImageFile(fh)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		job := &jobstore.Job{
			ID:           uuid.New().String(),
			Status:       jobstore.StatusQueued,
			ModelID:      req.ModelID,
			ScaleFactor:  req.ScaleFactor,
			OutputFormat: req.OutputFormat,
		}
		if err := h.Jobs.Create(c.Request.Context(), job); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "job kaydedilemedi: " + err.Error()})
			return
		}
		jobs = append(jobs, job)

		pbRequests = append(pbRequests, &pb.EnhanceRequest{
			Image: &pb.ImagePayload{
				Data:     data,
				MimeType: mimeType,
				Width:    width,
				Height:   height,
			},
			ModelId:      req.ModelID,
			ScaleFactor:  req.ScaleFactor,
			OutputFormat: req.OutputFormat,
		})
	}

	h.wg.Add(1)
	go func() {
		defer h.wg.Done()
		h.runEnhanceBatchJob(jobs, &pb.EnhanceBatchRequest{
			Requests:     pbRequests,
			MaxBatchSize: maxBatchImages,
		})
	}()

	resp := make([]gin.H, 0, len(jobs))
	for _, job := range jobs {
		resp = append(resp, gin.H{
			"job_id": job.ID,
			"status": string(job.Status),
		})
	}
	c.JSON(http.StatusAccepted, gin.H{"jobs": resp})
}

func (h *Handler) runEnhanceBatchJob(jobs []*jobstore.Job, req *pb.EnhanceBatchRequest) {
	ctx, cancel := context.WithTimeout(context.Background(), enhanceTimeout)
	defer cancel()

	for _, job := range jobs {
		job.Status = jobstore.StatusRunning
		_ = h.Jobs.Update(ctx, job)
	}

	if h.GRPCClient == nil {
		for _, job := range jobs {
			job.Status = jobstore.StatusFailed
			job.Error = "inference service bağlantısı yok (mock modda çalışılıyor)"
			_ = h.Jobs.Update(ctx, job)
			metrics.EnhanceJobsTotal.WithLabelValues(string(job.Status)).Inc()
		}
		return
	}

	resp, err := h.GRPCClient.EnhanceBatch(ctx, req)
	if err != nil {
		for _, job := range jobs {
			job.Status = jobstore.StatusFailed
			job.Error = err.Error()
			_ = h.Jobs.Update(ctx, job)
			metrics.EnhanceJobsTotal.WithLabelValues(string(job.Status)).Inc()
		}
		return
	}

	responses := resp.GetResponses()
	for i, job := range jobs {
		if i >= len(responses) {
			job.Status = jobstore.StatusFailed
			job.Error = "inference service yanıtı eksik"
		} else {
			applyEnhanceResponse(job, responses[i])
		}
		_ = h.Jobs.Update(ctx, job)
		metrics.EnhanceJobsTotal.WithLabelValues(string(job.Status)).Inc()
	}
}

// GetJob godoc
// @Summary      Get job status/result
// @Description  Reports the current state of a job queued by Enhance or EnhanceBatch. The result image (if any) is embedded as base64.
// @Tags         enhance
// @Produce      json
// @Param        job_id  path      string  true  "Job id"
// @Success      200     {object}  map[string]interface{}
// @Failure      404     {object}  map[string]interface{}
// @Router       /enhance/{job_id} [get]
func (h *Handler) GetJob(c *gin.Context) {
	jobID := c.Param("job_id")

	job, err := h.Jobs.Get(c.Request.Context(), jobID)
	if err != nil {
		if errors.Is(err, jobstore.ErrNotFound) {
			c.JSON(http.StatusNotFound, gin.H{"error": "job bulunamadı"})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	resp := gin.H{
		"job_id":        job.ID,
		"status":        string(job.Status),
		"model_id":      job.ModelID,
		"scale_factor":  job.ScaleFactor,
		"output_format": job.OutputFormat,
	}
	switch job.Status {
	case jobstore.StatusFailed:
		resp["error"] = job.Error
	case jobstore.StatusCompleted:
		resp["image"] = gin.H{
			"data_base64": base64.StdEncoding.EncodeToString(job.ResultImage),
			"mime_type":   job.ResultMime,
		}
		if job.Metrics != nil {
			resp["metrics"] = job.Metrics
		}
	}
	c.JSON(http.StatusOK, resp)
}

// respondBindError inspects a form/multipart binding error and responds
// with 413 if it was caused by the body exceeding the per-route size limit
// set in main.go (http.MaxBytesReader), or 400 otherwise.
func respondBindError(c *gin.Context, err error) {
	var tooLarge *http.MaxBytesError
	if errors.As(err, &tooLarge) {
		c.JSON(http.StatusRequestEntityTooLarge, gin.H{
			"error": "yüklenen dosya(lar) izin verilen boyutu aşıyor",
		})
		return
	}
	c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
}

// readImageFile reads a multipart file into memory and derives its MIME
// type and pixel dimensions. Dimensions are best-effort: formats the
// standard library doesn't decode (e.g. FITS) are still accepted, just
// without width/height.
func readImageFile(fh *multipart.FileHeader) (data []byte, mimeType string, width, height uint32, err error) {
	src, err := fh.Open()
	if err != nil {
		return nil, "", 0, 0, errors.New("dosya okunamadı")
	}
	defer src.Close()

	data, err = io.ReadAll(src)
	if err != nil {
		return nil, "", 0, 0, errors.New("dosya okunamadı")
	}

	mimeType = http.DetectContentType(data)
	if cfg, _, decErr := image.DecodeConfig(bytes.NewReader(data)); decErr == nil {
		width = uint32(cfg.Width)
		height = uint32(cfg.Height)
	}

	return data, mimeType, width, height, nil
}
