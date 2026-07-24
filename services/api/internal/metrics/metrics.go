// Package metrics exposes Prometheus collectors for the API gateway and a
// gin middleware that records them on every request.
package metrics

import (
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	HTTPRequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "api_http_requests_total",
		Help: "Total HTTP requests handled by the API gateway.",
	}, []string{"method", "path", "status"})

	HTTPRequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "api_http_request_duration_seconds",
		Help:    "HTTP request latency in seconds.",
		Buckets: prometheus.DefBuckets,
	}, []string{"method", "path"})

	EnhanceJobsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "api_enhance_jobs_total",
		Help: "Total /enhance jobs by final status.",
	}, []string{"status"})
)

// Middleware records request count and latency for every request. Register
// it before the route handlers so it wraps the full request lifecycle.
func Middleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		c.Next()

		path := c.FullPath()
		if path == "" {
			path = "unmatched"
		}
		status := strconv.Itoa(c.Writer.Status())

		HTTPRequestsTotal.WithLabelValues(c.Request.Method, path, status).Inc()
		HTTPRequestDuration.WithLabelValues(c.Request.Method, path).Observe(time.Since(start).Seconds())
	}
}

// Handler serves the Prometheus text exposition format for scraping.
func Handler() http.Handler {
	return promhttp.Handler()
}
