"""Generated proto stubs for the DeepHorizon inference server.

This package contains the Python bindings produced by ``grpc_tools.protoc``
from ``proto/deephorizon/v1/*.proto``. The stubs are consumed by
``services.ml.inference_server.server`` to implement the
``InferenceService`` gRPC contract.

Modules:
    deephorizon.v1.common_pb2: ImagePayload, Metrics, JobStatus.
    deephorizon.v1.inference_pb2: EnhanceRequest/Response, ListModels, Health.
    deephorizon.v1.inference_pb2_grpc: gRPC servicer and stub classes.

Regenerate with::

    cd proto && buf generate

Or, if ``buf`` is not installed::

    python -m grpc_tools.protoc \\
        --proto_path=proto \\
        --python_out=services/ml/inference_server/pb \\
        --grpc_python_out=services/ml/inference_server/pb \\
        proto/deephorizon/v1/common.proto \\
        proto/deephorizon/v1/inference.proto
"""

from __future__ import annotations

from deephorizon.v1 import (
    common_pb2,
    inference_pb2,
    inference_pb2_grpc,
)

__all__ = ["common_pb2", "inference_pb2", "inference_pb2_grpc"]
