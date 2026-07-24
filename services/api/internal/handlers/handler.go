package handlers

import (
	"sync"
	"time"

	grpcclient "github.com/Octapull/deephorizon/services/api/internal/grpc_client"
	"github.com/Octapull/deephorizon/services/api/internal/jobstore"
)

type Handler struct {
	GRPCClient *grpcclient.Client
	Jobs       *jobstore.Store

	// wg tracks in-flight background job goroutines (see enhance.go) so
	// shutdown can drain them before closing GRPCClient/Jobs out from
	// under them.
	wg sync.WaitGroup
}

func New(grpcClient *grpcclient.Client, jobs *jobstore.Store) *Handler {
	return &Handler{
		GRPCClient: grpcClient,
		Jobs:       jobs,
	}
}

// Wait blocks until all in-flight background jobs finish or timeout
// elapses, whichever comes first. Call during shutdown, after the HTTP
// server has stopped accepting requests but before closing GRPCClient and
// the Redis connection those goroutines write to.
func (h *Handler) Wait(timeout time.Duration) {
	done := make(chan struct{})
	go func() {
		h.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(timeout):
	}
}
