// Package jobstore persists async /enhance job state in Redis so the
// gateway can accept a request, return immediately, and let the caller
// poll GET /enhance/:job_id for the result.
package jobstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

type Status string

const (
	StatusQueued    Status = "JOB_STATUS_QUEUED"
	StatusRunning   Status = "JOB_STATUS_RUNNING"
	StatusCompleted Status = "JOB_STATUS_COMPLETED"
	StatusFailed    Status = "JOB_STATUS_FAILED"
)

// ErrNotFound is returned by Get when the job id does not exist or has
// expired (jobs are stored with a TTL, see New).
var ErrNotFound = errors.New("jobstore: job not found")

type Metrics struct {
	PSNR            float64 `json:"psnr"`
	SSIM            float64 `json:"ssim"`
	LPIPS           float64 `json:"lpips"`
	InferenceTimeMs uint32  `json:"inference_time_ms"`
}

type Job struct {
	ID           string    `json:"job_id"`
	Status       Status    `json:"status"`
	ModelID      string    `json:"model_id"`
	ScaleFactor  uint32    `json:"scale_factor"`
	OutputFormat string    `json:"output_format"`
	Error        string    `json:"error,omitempty"`
	ResultImage  []byte    `json:"result_image,omitempty"`
	ResultMime   string    `json:"result_mime,omitempty"`
	Metrics      *Metrics  `json:"metrics,omitempty"`
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"updated_at"`
}

type Store struct {
	rdb *redis.Client
	ttl time.Duration
}

// New wraps an existing Redis client. ttl controls how long a job's state
// survives after being written — long enough for a client to poll it, short
// enough that Redis doesn't accumulate stale jobs forever.
func New(rdb *redis.Client, ttl time.Duration) *Store {
	return &Store{rdb: rdb, ttl: ttl}
}

func key(id string) string {
	return fmt.Sprintf("job:%s", id)
}

func (s *Store) Create(ctx context.Context, job *Job) error {
	now := time.Now().UTC()
	job.CreatedAt = now
	job.UpdatedAt = now
	return s.save(ctx, job)
}

func (s *Store) Update(ctx context.Context, job *Job) error {
	job.UpdatedAt = time.Now().UTC()
	return s.save(ctx, job)
}

func (s *Store) save(ctx context.Context, job *Job) error {
	data, err := json.Marshal(job)
	if err != nil {
		return fmt.Errorf("jobstore: marshal job %s: %w", job.ID, err)
	}
	if err := s.rdb.Set(ctx, key(job.ID), data, s.ttl).Err(); err != nil {
		return fmt.Errorf("jobstore: write job %s: %w", job.ID, err)
	}
	return nil
}

func (s *Store) Get(ctx context.Context, id string) (*Job, error) {
	data, err := s.rdb.Get(ctx, key(id)).Bytes()
	if errors.Is(err, redis.Nil) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, fmt.Errorf("jobstore: read job %s: %w", id, err)
	}
	var job Job
	if err := json.Unmarshal(data, &job); err != nil {
		return nil, fmt.Errorf("jobstore: unmarshal job %s: %w", id, err)
	}
	return &job, nil
}
