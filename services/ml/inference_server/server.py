"""gRPC inference server for DeepHorizon models.

Implements the ``InferenceService`` proto contract defined in
``proto/deephorizon/v1/inference.proto``. The server:

1. Loads ONNX models via the ``ModelRegistry``
2. Decodes incoming ``ImagePayload`` bytes to tensors
3. Runs model inference
4. Encodes output tensors back to the requested format
5. Returns ``EnhanceResponse`` with image + metrics

RPCs implemented:
    - ``Enhance``: single image enhancement
    - ``EnhanceBatch``: batch enhancement (sequential)
    - ``ListModels``: enumerate registered models
    - ``Health``: server health check

Usage:
    python -m services.ml.inference_server.server \\
        --onnx-path exports/deephorizon_generator.onnx \\
        --model-id pix2pix-v1 \\
        --port 50051

Note:
    The generated proto stubs (``inference_server_pb2``,
    ``inference_server_pb2_grpc``) are expected to be in the same
    directory. Generate them with:
        cd proto && buf generate
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np

from services.ml.inference_server.image_utils import (
    decode_image_to_tensor,
    encode_tensor_to_image,
)
from services.ml.inference_server.metrics import (
    record_batch_size,
    record_inference,
)
from services.ml.inference_server.model_registry import ModelRegistry, ModelSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Proto imports — generated stubs (created by `buf generate`)
# ---------------------------------------------------------------------------


def _import_proto_stubs():
    """Import generated proto stubs.

    Returns:
        Tuple of ``(inference_pb2, inference_pb2_grpc)``.

    Raises:
        ImportError: If stubs haven't been generated yet.
    """
    try:
        from services.ml.inference_server.pb import (  # type: ignore[import-untyped]
            inference_pb2,
            inference_pb2_grpc,
        )

        return inference_pb2, inference_pb2_grpc
    except ImportError as exc:
        raise ImportError(
            "Proto stubs not found. Generate them with:\n"
            "  cd proto && buf generate\n"
            "Or, with grpc_tools.protoc:\n"
            "  python -m grpc_tools.protoc --proto_path=proto \\\n"
            "    --python_out=services/ml/inference_server/pb \\\n"
            "    --grpc_python_out=services/ml/inference_server/pb \\\n"
            "    proto/deephorizon/v1/common.proto \\\n"
            "    proto/deephorizon/v1/inference.proto"
        ) from exc


# ---------------------------------------------------------------------------
# Servicer
# ---------------------------------------------------------------------------


class InferenceServicer:
    """Implementation of the ``InferenceService`` gRPC service.

    Args:
        registry: ModelRegistry with pre-registered ONNX models.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # Enhance (single image)
    # ------------------------------------------------------------------
    def Enhance(self, request, context):  # noqa: N802 (gRPC naming)
        """Enhance a single image."""
        inference_pb2, _ = _import_proto_stubs()

        # Resolve model_id early for metrics labeling
        model_id_for_metrics = request.model_id or "unknown"
        start_time = time.perf_counter()

        try:
            # Validate request
            if not request.image.data:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("image.data is empty")
                record_inference(
                    model_id=model_id_for_metrics,
                    status="invalid_argument",
                    latency_seconds=time.perf_counter() - start_time,
                )
                return inference_pb2.EnhanceResponse()

            model_id = request.model_id or self._default_model_id(context)
            if model_id is None:
                record_inference(
                    model_id=model_id_for_metrics,
                    status="not_found",
                    latency_seconds=time.perf_counter() - start_time,
                )
                return inference_pb2.EnhanceResponse()

            model_id_for_metrics = model_id

            # Decode input
            input_tensor = decode_image_to_tensor(
                data=request.image.data,
                mime_type=request.image.mime_type or "image/png",
                width=request.image.width or None,
                height=request.image.height or None,
            )

            # Run inference
            inference_start = time.perf_counter()
            output_array = self._registry.run_inference(
                model_id=model_id,
                input_tensor=input_tensor.numpy(),
            )
            inference_time_ms = int((time.perf_counter() - inference_start) * 1000)

            # Encode output
            output_format = request.output_format or "png"
            output_tensor = _numpy_to_tensor(output_array)
            output_bytes, output_mime, out_w, out_h = encode_tensor_to_image(
                output_tensor,
                output_format=output_format,
            )

            # Record success metrics
            record_inference(
                model_id=model_id,
                status="success",
                latency_seconds=time.perf_counter() - start_time,
            )

            # Build response
            return inference_pb2.EnhanceResponse(
                image=inference_pb2.ImagePayload(
                    data=output_bytes,
                    mime_type=output_mime,
                    width=out_w,
                    height=out_h,
                ),
                metrics=inference_pb2.Metrics(
                    psnr=0.0,  # No ground truth in inference
                    ssim=0.0,
                    lpips=0.0,
                    inference_time_ms=inference_time_ms,
                ),
                model_id=model_id,
                status=inference_pb2.JOB_STATUS_COMPLETED,
            )

        except KeyError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            record_inference(
                model_id=model_id_for_metrics,
                status="not_found",
                latency_seconds=time.perf_counter() - start_time,
            )
            return inference_pb2.EnhanceResponse()
        except Exception as exc:
            logger.exception("Enhance failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Enhance failed: {exc}")
            record_inference(
                model_id=model_id_for_metrics,
                status="error",
                latency_seconds=time.perf_counter() - start_time,
            )
            return inference_pb2.EnhanceResponse()

    # ------------------------------------------------------------------
    # EnhanceBatch
    # ------------------------------------------------------------------
    def EnhanceBatch(self, request, context):  # noqa: N802
        """Enhance a batch of images sequentially."""
        inference_pb2, _ = _import_proto_stubs()

        # Record batch size for the first model_id in the batch
        batch_size = len(request.requests)
        if batch_size > 0 and request.requests[0].model_id:
            record_batch_size(
                model_id=request.requests[0].model_id,
                batch_size=batch_size,
            )

        responses = []
        for req in request.requests:
            # Reuse single-image logic
            response = self.Enhance(req, context)
            responses.append(response)

        return inference_pb2.EnhanceBatchResponse(responses=responses)

    # ------------------------------------------------------------------
    # ListModels
    # ------------------------------------------------------------------
    def ListModels(self, request, context):  # noqa: N802
        """List all registered models."""
        inference_pb2, _ = _import_proto_stubs()

        models = []
        for spec in self._registry.list_specs():
            models.append(
                inference_pb2.ModelInfo(
                    id=spec.model_id,
                    architecture=spec.architecture,
                    version=spec.version,
                    validation_metrics=inference_pb2.Metrics(
                        psnr=spec.psnr,
                        ssim=spec.ssim,
                    ),
                )
            )

        return inference_pb2.ListModelsResponse(models=models)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def Health(self, request, context):  # noqa: N802
        """Server health check."""
        inference_pb2, _ = _import_proto_stubs()

        try:
            import torch  # noqa: F401

            gpu_available = torch.cuda.is_available()
        except ImportError:
            gpu_available = False

        loaded_count = len(self._registry.list_loaded())
        registered_count = len(self._registry.list_specs())

        return inference_pb2.HealthResponse(
            ok=True,
            detail=f"{registered_count} models registered, {loaded_count} loaded",
            gpu_available=gpu_available,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _default_model_id(self, context) -> str | None:
        """Return the first registered model_id, or set NOT_FOUND."""
        specs = self._registry.list_specs()
        if not specs:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("No models registered")
            return None
        return specs[0].model_id


def _numpy_to_tensor(array: np.ndarray):
    """Convert numpy array to torch tensor (lazy import)."""
    import torch

    return torch.from_numpy(array.copy())


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------


def build_registry_from_args(args: argparse.Namespace) -> ModelRegistry:
    """Build a ModelRegistry from CLI arguments.

    Supports two modes:
    - ``--onnx-path``: register a single model
    - ``--models-dir``: register all ``.onnx`` files in a directory
    """
    registry = ModelRegistry()

    if args.onnx_path:
        spec = ModelSpec(
            model_id=args.model_id or "default",
            architecture=args.architecture,
            version=args.version,
            onnx_path=Path(args.onnx_path),
            metadata_path=(
                Path(args.onnx_path + ".json")
                if Path(args.onnx_path + ".json").exists()
                else None
            ),
        )
        registry.register(spec)

    if args.models_dir:
        models_dir = Path(args.models_dir)
        for onnx_file in sorted(models_dir.glob("*.onnx")):
            model_id = onnx_file.stem
            metadata_file = onnx_file.with_suffix(onnx_file.suffix + ".json")
            spec = ModelSpec(
                model_id=model_id,
                architecture=args.architecture,
                version=args.version,
                onnx_path=onnx_file,
                metadata_path=metadata_file if metadata_file.exists() else None,
            )
            try:
                registry.register(spec)
            except FileNotFoundError:
                logger.warning("Skipping missing model: %s", onnx_file)

    return registry


def serve(
    registry: ModelRegistry,
    port: int = 50051,
    max_workers: int = 4,
) -> grpc.Server:
    """Start the gRPC server.

    Args:
        registry: ModelRegistry with pre-registered models.
        port: TCP port to bind.
        max_workers: Thread pool size for concurrent requests.

    Returns:
        The running gRPC server (call ``server.stop()`` to shut down).
    """
    _, inference_pb2_grpc = _import_proto_stubs()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(registry), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("Inference server started on port %d", port)
    return server


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="DeepHorizon gRPC inference server")
    parser.add_argument(
        "--onnx-path",
        type=str,
        default=None,
        help="Path to a single ONNX model file",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory containing multiple .onnx files",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model ID for --onnx-path (default: 'default')",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="pix2pix",
        help="Model architecture (default: pix2pix)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Model version (default: 1.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="TCP port (default: 50051)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Thread pool size (default: 4)",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=8000,
        help="Prometheus metrics HTTP port (default: 8000)",
    )
    parser.add_argument(
        "--disable-metrics",
        action="store_true",
        help="Disable Prometheus metrics server",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Prometheus metrics server başlat
    if not args.disable_metrics:
        from services.ml.inference_server.metrics import start_metrics_server

        start_metrics_server(port=args.metrics_port)

    registry = build_registry_from_args(args)
    if not registry.list_specs():
        logger.error("No models registered. Use --onnx-path or --models-dir.")
        return

    server = serve(registry, port=args.port, max_workers=args.max_workers)
    logger.info("Registered models: %s", [s.model_id for s in registry.list_specs()])

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop(grace=5)


if __name__ == "__main__":
    main()
