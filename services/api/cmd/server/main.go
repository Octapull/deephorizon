package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"

	grpcclient "github.com/Octapull/deephorizon/services/api/internal/grpc_client"
	"github.com/Octapull/deephorizon/services/api/internal/handlers"
	"github.com/Octapull/deephorizon/services/api/internal/jobstore"
	apimetrics "github.com/Octapull/deephorizon/services/api/internal/metrics"
)

const (
	// jobTTL controls how long a completed/failed job's state survives in
	// Redis — long enough for a client to poll it after the response lands.
	jobTTL = 24 * time.Hour

	// shutdownTimeout bounds how long the HTTP server waits for in-flight
	// requests to finish once a shutdown signal arrives.
	shutdownTimeout = 15 * time.Second

	// jobDrainTimeout bounds how long we wait for background /enhance
	// goroutines (see internal/handlers.Handler.wg) to finish before
	// closing the gRPC client and Redis connection they depend on. Kept
	// short relative to enhanceTimeout (5m) because it's bounded by
	// Kubernetes' default terminationGracePeriodSeconds (30s) — a job
	// still running past this point is killed anyway when the pod dies.
	jobDrainTimeout = 20 * time.Second

	defaultMaxEnhanceMB = 25  // single image, /enhance
	defaultMaxBatchMB   = 200 // up to 10 images, /enhance/batch
)

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getenvMB(key string, fallbackMB int64) int64 {
	v := os.Getenv(key)
	if v == "" {
		return fallbackMB << 20
	}
	mb, err := strconv.ParseInt(v, 10, 64)
	if err != nil || mb <= 0 {
		log.Printf("%s geçersiz (%q), varsayılan %dMB kullanılıyor", key, v, fallbackMB)
		return fallbackMB << 20
	}
	return mb << 20
}

// limitBody caps the request body at maxBytes. Handlers surface the
// resulting error as 413 Payload Too Large — see respondBindError in
// internal/handlers/enhance.go.
func limitBody(maxBytes int64) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxBytes)
		c.Next()
	}
}

// @title           DeepHorizon API Gateway
// @version         1.0
// @description     REST gateway in front of the Python gRPC inference service. Translates multipart image uploads into Enhance/EnhanceBatch RPCs and tracks job state in Redis.
// @BasePath        /
func main() {
	grpcAddr := getenv("GRPC_ADDR", "localhost:50051")
	grpcClient, err := grpcclient.New(grpcAddr)
	if err != nil {
		log.Printf("gRPC bağlantısı kurulamadı (mock modda çalışılıyor): %v", err)
	}

	rdb := redis.NewClient(&redis.Options{
		Addr:     getenv("REDIS_ADDR", "localhost:6379"),
		Password: os.Getenv("REDIS_PASSWORD"),
		DB:       0,
	})

	pingCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	if err := rdb.Ping(pingCtx).Err(); err != nil {
		log.Printf("Redis'e bağlanılamadı, job store çalışmayacak: %v", err)
	}
	cancel()

	jobs := jobstore.New(rdb, jobTTL)
	h := handlers.New(grpcClient, jobs)

	allowedOrigins := strings.Split(getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000"), ",")
	maxEnhanceBytes := getenvMB("MAX_ENHANCE_MB", defaultMaxEnhanceMB)
	maxBatchBytes := getenvMB("MAX_BATCH_MB", defaultMaxBatchMB)

	r := gin.Default()
	r.Use(apimetrics.Middleware())
	r.Use(cors.New(cors.Config{
		AllowOrigins:     allowedOrigins,
		AllowMethods:     []string{"GET", "POST", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept"},
		AllowCredentials: false,
		MaxAge:           12 * time.Hour,
	}))

	r.GET("/health", h.Health)
	r.GET("/models", h.ListModels)
	r.GET("/models/:id", h.GetModel)
	r.POST("/enhance", limitBody(maxEnhanceBytes), h.Enhance)
	r.POST("/enhance/batch", limitBody(maxBatchBytes), h.EnhanceBatch)
	r.GET("/enhance/:job_id", h.GetJob)
	r.GET("/metrics", gin.WrapH(apimetrics.Handler()))

	srv := &http.Server{
		Addr:    ":8080",
		Handler: r,
	}

	go func() {
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("sunucu başlatılamadı: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("kapatma sinyali alındı, sunucu düzgün şekilde kapatılıyor...")

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer shutdownCancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("sunucu düzgün kapatılamadı: %v", err)
	}

	// Let in-flight background /enhance jobs finish before yanking Redis
	// and the gRPC client out from under them.
	h.Wait(jobDrainTimeout)

	if grpcClient != nil {
		grpcClient.Close()
	}
	rdb.Close()

	log.Println("sunucu kapatıldı")
}
